import sys
import os
import warnings

# adding 2.0-RecModules folder in sys.path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


import numpy as np
import copy
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
import scipy
from sklearn.preprocessing import normalize
import time
from scipy.stats import beta

import csv
import math
from collections import Counter

from recbole.data.utils import *
from utils.model_utils import load_model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def safe_entropy(p, eps=1e-12):
    """Shannon entropy of a probability vector (natural log)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + eps)).sum())

def category_entropy_from_items(items_zero_based, item_label_dict):
    """Compute entropy over categories for a list/array of item indices (0-based)."""
    if item_label_dict is None:
        return None
    cats = [item_label_dict.get(int(it), None) for it in items_zero_based]
    cats = [c for c in cats if c is not None]
    if len(cats) == 0:
        return 0.0
    uniq, counts = np.unique(cats, return_counts=True)
    p = counts / counts.sum()
    return safe_entropy(p)

def diversity_rerank_topk(items_1based, scores_probs, item_label_dict, lambda_diversity=0.3):
    """Greedy rerank to increase category diversity while keeping high-score items early.

    items_1based: numpy array of shape (topk,)
    scores_probs: numpy array of shape (topk,) (not necessarily normalized)
    """
    if item_label_dict is None or lambda_diversity <= 0:
        return items_1based, scores_probs

    items_1based = np.asarray(items_1based).copy()
    scores_probs = np.asarray(scores_probs).copy()

    # Map to categories; unknown category treated as its own bucket
    cats = [item_label_dict.get(int(it - 1), "UNK") for it in items_1based]

    selected = []
    selected_scores = []
    seen_cats = set()

    # Normalize scores to [0,1] for stable mixing
    s = scores_probs.astype(float)
    if np.max(s) > np.min(s):
        s_norm = (s - np.min(s)) / (np.max(s) - np.min(s))
    else:
        s_norm = np.ones_like(s)

    remaining = list(range(len(items_1based)))
    while remaining:
        best_idx = None
        best_val = -1e18
        for ridx in remaining:
            cat = cats[ridx]
            novelty = 1.0 if cat not in seen_cats else 0.0
            val = (1 - lambda_diversity) * s_norm[ridx] + lambda_diversity * novelty
            if val > best_val:
                best_val = val
                best_idx = ridx
        selected.append(items_1based[best_idx])
        selected_scores.append(scores_probs[best_idx])
        seen_cats.add(cats[best_idx])
        remaining.remove(best_idx)

    return np.array(selected), np.array(selected_scores)

def init_metrics_file(metrics_path):
    if not os.path.exists(os.path.dirname(metrics_path)):
        os.makedirs(os.path.dirname(metrics_path))
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("iter_b,step_d,user,topk_cat_entropy,history_cat_entropy,chosen_item,chosen_cat\\n")


def category_entropy(items_1idx, item_label_dict):
    """
    items_1idx: list/array of item ids in [1..num_items]
    item_label_dict: maps item 0-index -> category label
    """
    if items_1idx is None or len(items_1idx) == 0 or item_label_dict is None:
        return 0.0

    cats = []
    for it in items_1idx:
        it0 = int(it) - 1
        if it0 in item_label_dict:
            cats.append(item_label_dict[it0])

    if len(cats) == 0:
        return 0.0

    counts = Counter(cats)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)

    return ent


def normalized_entropy(items_1idx, item_label_dict):
    """
    Normalized entropy in [0..1]. 0 => single-category bubble, 1 => maximally diverse.
    """
    if items_1idx is None or len(items_1idx) == 0 or item_label_dict is None:
        return 0.0

    cats = []
    for it in items_1idx:
        it0 = int(it) - 1
        if it0 in item_label_dict:
            cats.append(item_label_dict[it0])

    unique = len(set(cats))
    if unique <= 1:
        return 0.0

    ent = category_entropy(items_1idx, item_label_dict)
    return ent / (math.log(unique + 1e-12))


def mmr_diversity_rerank(user_recs_1idx, scores_probs, item_label_dict, lambda_div=0.5):
    """
    Re-rank Top-K to penalize repeated categories.
    - user_recs_1idx: array/list of item ids in [1..num_items]
    - scores_probs: probabilities aligned with user_recs_1idx
    - lambda_div: 0..1 (higher => more diversity pressure)
    Returns: reordered np.array of item ids (same length)
    """
    recs = list(user_recs_1idx)
    probs = list(scores_probs)

    # relevance-first initial order
    candidates = sorted(zip(recs, probs), key=lambda x: float(x[1]), reverse=True)

    selected = []
    seen_cats = set()

    while len(candidates) > 0:
        best_idx = 0
        best_score = -1e18

        for idx, (it, p) in enumerate(candidates):
            cat = item_label_dict.get(int(it) - 1, None)
            penalty = 1.0 if (cat is not None and cat in seen_cats) else 0.0
            score = (1.0 - lambda_div) * float(p) - lambda_div * penalty

            if score > best_score:
                best_score = score
                best_idx = idx

        it, p = candidates.pop(best_idx)
        selected.append(int(it))
        cat = item_label_dict.get(int(it) - 1, None)
        if cat is not None:
            seen_cats.add(cat)

    return np.array(selected, dtype=np.int64)


def normalize_T(T, B):
    T /= B
    new_matrix = normalize(T, norm="l1", axis=1)
    new_matrix = np.around(new_matrix, decimals=2)

    return new_matrix


def retrieve_parameters(T, reverse_videos_dict):
    indexes = T.nonzero()

    edges_index = []
    edges_weight = []
    nodes = []

    for i, j in zip(indexes[0], indexes[1]):

        reindexed_i = reverse_videos_dict[i]
        reindexed_j = reverse_videos_dict[j]

        edges_index.append((reindexed_i, reindexed_j))
        edges_weight.append(T[i, j])

        if reindexed_i not in nodes:
            nodes.append(reindexed_i)
        if reindexed_j not in nodes:
            nodes.append(reindexed_j)

    return nodes, edges_index, edges_weight


def create_graph_tsv(
        nodes,
        edges_index,
        edges_weight,
        filename,
        reverse_videos_labels_dict):


    f = open(filename + "_edge.tsv", "w")
    f.write("Source\tTarget\tWeight\n")
    for i in range(len(edges_index)):
        f.write(
            "{}\t{}\t{}\n".format(
                edges_index[i][0],
                edges_index[i][1],
                edges_weight[i]))
    f.close()

    f = open(filename + "_node.tsv", "w")
    f.write("Id\tLabel\tCategory\n")
    for i in range(len(nodes)):
        category = reverse_videos_labels_dict[nodes[i]]

        f.write("{}\t{}\t{}\n".format(nodes[i], nodes[i], category))

    f.close()
    print("saved", filename)


def init_T(sess, num_items):
    n = num_items + 1
    T = lil_matrix((n, n), dtype=np.float64)

    for i in range(len(sess) - 1):
        T[sess[i], sess[i + 1]] = 1

    return T.tocsr()


def create_T_tensor(
        history_dataset,
        num_items=0):

    T_tensor = []

    count = 0

    for user, history in list(history_dataset.items()):
        T = init_T(history, num_items)
        T_tensor.append(T)

        if count % 100 == 0:
            print("Done", count)

        count += 1

    return copy.deepcopy(T_tensor)


def nullify_history_scores(temp_histories, scores):
    for user, history in enumerate(temp_histories):
        scores[user, history] = -1



def simulate_organic_model_with_alpha(X, Y, noise_cov, histories, n_trials=30, shrinkage_alpha=None, verbosity=1):
    """
    Simulate organic model with alpha that controls shrinkage level.
    """
    n_users, n_dim = X.shape
    # print(n_users)
    assert Y.shape[1] == n_dim
    n_movies = Y.shape[0]
    assert noise_cov.shape == (n_dim, n_dim)
    noise_mean = np.zeros(n_dim)
    user_choices_per_trial = np.zeros((n_users, n_trials))

    if shrinkage_alpha is not None:
        assert shrinkage_alpha > 0 and shrinkage_alpha <= 1
        print('Running simulation with shrinkage alpha =', shrinkage_alpha)

    rng = np.random.default_rng()
    for t in range(n_trials):

        if verbosity > 0 and t % verbosity == 0:
            print('trial', t)

        # noisy_t = time.time()
        noisy_movies = Y + rng.multivariate_normal(noise_mean, noise_cov, (n_users, n_movies))

        # for_t = time.time()
        for u in range(n_users):  # need to go user by user bc each user has their own noisy sample of movies

            if shrinkage_alpha is not None:
                noisy_mean = np.mean(noisy_movies[u], axis=0)
                noisy_movies[u] = ((1 - shrinkage_alpha) * noisy_movies[u]) + (shrinkage_alpha * noisy_mean)

            noisy_dists = cdist([X[u]], noisy_movies[u], 'cosine')
            assert noisy_dists.shape == (1, n_movies)

            noisy_dists[0][histories[u]] = 10000 # nullify # max(noisy_dists[0]) + 1
            # if u == 0:
            #     print(histories[0])
            user_choice = np.argmin(noisy_dists[0])  # index of movie chosen by user
            user_choices_per_trial[u, t] = user_choice
            histories[u].append(user_choice)

    return user_choices_per_trial


# Organic Simulation
def generate_organic_graphs(T_tensor, history_dataset, name, dataset_path, B, d,
                            reverse_users_dict, reverse_videos_dict, reverse_videos_labels_dict,
                            saving_path):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    warnings.simplefilter(
        'ignore',
        category=scipy.sparse.SparseEfficiencyWarning)

    saving_path += "a_0.4/"
    graphs_folder = saving_path + "graphs/"

    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)

    users_embeddings_fn = dataset_path + "/Users_embeddings.npy"
    items_embeddings_fn = dataset_path + "/Items_embeddings.npy"

    users_embeddings = np.load(users_embeddings_fn, allow_pickle=True)
    items_embeddings = np.load(items_embeddings_fn, allow_pickle=True)

    # debug
    videos_dict = {v: k for k, v in reverse_videos_dict.items()}

    new_items_embeddings = np.ndarray((items_embeddings.shape[0], 2), object)
    for item_emb in items_embeddings:
        item_real_idx = item_emb[0]
        emb = item_emb[1]

        item_reindexed = videos_dict[item_real_idx]

        new_items_embeddings[item_reindexed][0] = item_reindexed
        new_items_embeddings[item_reindexed][1] = emb

    items_embeddings = new_items_embeddings
    # end debug

    X = np.stack(users_embeddings[:, 1])
    Y = np.stack(items_embeddings[:, 1])

    noise_cov = np.cov(Y, rowvar=False) * 0.5
    # torch.cov(torch.FloatTensor(Y).T) * 0.5

    # noise_cov = torch.mm(noise_cov, noise_cov.t())
    # noise_cov.add_(torch.eye(noise_cov.shape[0]))

    shrinkage_alpha = 0.4

    org_alpha2results = {}

    histories = np.array(list(history_dataset.values()))

    for b in range(B):
        print('B =', b)
        start = time.time()
        histories_copy = copy.deepcopy(histories)
        choices = simulate_organic_model_with_alpha(X, Y, noise_cov, histories_copy,
                                                    shrinkage_alpha=shrinkage_alpha, verbosity=20, n_trials=d)

        print("End", b, "in", time.time() - start)
        # break
        org_alpha2results[b] = choices

    for b, users_choices in org_alpha2results.items():
        for i in range(len(users_choices)):
            user_choices = users_choices[i]
            curr_item = history_dataset[i][-1]
            for j, item_index in enumerate(user_choices):  # delta loop
                item_index = int(item_index)
                reindexed_item = item_index

                T_tensor[i][curr_item, reindexed_item] += 1

                curr_item = reindexed_item


    for i in range(len(T_tensor)):
        T_tensor[i] = normalize_T(T_tensor[i], B)

        nodes, edges, weights = retrieve_parameters(
            T_tensor[i], reverse_videos_dict)

        filename = name + "_" + str(reverse_users_dict[i])
        path_file = graphs_folder + filename
        # print(path_file)
        create_graph_tsv(nodes, edges, weights, path_file, reverse_videos_labels_dict)

def generate_organic_preferences(X, Y, shrinkage_alpha, noise_cov):

    n_users, n_dim = X.shape
    # print(n_users)
    assert Y.shape[1] == n_dim
    n_items = Y.shape[0]
    assert noise_cov.shape == (n_dim, n_dim)
    noise_mean = np.zeros(n_dim)

    if shrinkage_alpha is not None:
        assert shrinkage_alpha > 0 and shrinkage_alpha <= 1

    rng = np.random.default_rng()
    noisy_items = Y + rng.multivariate_normal(noise_mean, noise_cov, (n_users, n_items))
    organic_preferences = []

    for u in range(n_users):  # need to go user by user bc each user has their own noisy sample of items

        if shrinkage_alpha is not None:
            noisy_mean = np.mean(noisy_items[u], axis=0)
            noisy_items[u] = ((1 - shrinkage_alpha) * noisy_items[u]) + (shrinkage_alpha * noisy_mean)

        noisy_dists = cdist([X[u]], noisy_items[u], 'cosine')
        assert noisy_dists.shape == (1, n_items)

        noisy_dists[0] = 1-noisy_dists[0] # we need similarities
        organic_preferences.append(noisy_dists[0])

    organic_preferences = np.array(organic_preferences)
    return organic_preferences

def prepare_organic_model(dataset_path, reverse_videos_dict):

    users_embeddings_fn = dataset_path + "/Users_embeddings.npy"
    items_embeddings_fn = dataset_path + "/Items_embeddings.npy"

    users_embeddings = np.load(users_embeddings_fn, allow_pickle=True)
    items_embeddings = np.load(items_embeddings_fn, allow_pickle=True)

    videos_dict = {v: k for k, v in reverse_videos_dict.items()}

    new_items_embeddings = np.ndarray((items_embeddings.shape[0], 2), object)
    for item_emb in items_embeddings:

        item_real_idx = item_emb[0]
        emb = item_emb[1]

        item_reindexed = videos_dict[item_real_idx]

        new_items_embeddings[item_reindexed][0] = item_reindexed
        new_items_embeddings[item_reindexed][1] = emb

    items_embeddings = new_items_embeddings

    X = np.stack(users_embeddings[:, 1])
    Y = np.stack(items_embeddings[:, 1])

    noise_cov = np.cov(Y, rowvar=False) * 0.5

    shrinkage_alpha = 0.4

    return X, Y, shrinkage_alpha, noise_cov


def prepare_organic_model_movielens(history_dataset, item_label_dict):
    categories = np.unique(list(item_label_dict.values()))
    num_categories = len(categories)
    num_items = len(np.unique(list(item_label_dict.keys())))
    organic_preferences = np.zeros((len(history_dataset), num_items))

    for u in range(len(history_dataset)):
        history = history_dataset[u]
        categories_preferences = np.zeros(num_categories)
        for h in history:
            for idx, cat in enumerate(categories):
                if cat == item_label_dict[h]:
                    break
            categories_preferences[idx] += 1
        sum_values = np.sum(categories_preferences)
        for idx, cat in enumerate(categories):
            categories_preferences[idx] /= sum_values
        for i in range(num_items):
            for idx, cat in enumerate(categories):
                if cat == item_label_dict[i]:
                    break
            organic_preferences[u][i] = categories_preferences[idx]

    return organic_preferences


def generate_organic_graphs_movielens(history_dataset, item_label_dict):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    warnings.simplefilter(
        'ignore',
        category=scipy.sparse.SparseEfficiencyWarning)

    saving_path += "a_0.4/"
    graphs_folder = saving_path + "graphs/"

    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)

    histories = np.array(list(history_dataset.values()))

    organic_preferences = prepare_organic_model_movielens(history_dataset, item_label_dict)

    for iter_b in range(B):
        print('B =', iter_b)
        start = time.time()
        histories_copy = copy.deepcopy(histories)
        for j in range(d):
            organic_user_preferences = copy.deepcopy(organic_preferences[i])
            organic_user_preferences[temp_histories[i] - 1] = -10000
            item_sampled = np.argmax(organic_preferences_user_tmp) + 1  # +1 to be consistent

            if np.sum(organic_user_preferences) > 0:
                organic_user_preferences /= np.sum(organic_user_preferences)

            item_sampled = np.random.choice(
                topk_recommendations[i].detach().cpu().numpy(), p=organic_user_preferences)

        print("End", iter_b, "in", time.time() - start)
        # break
        org_alpha2results[iter_b] = choices

    for b, users_choices in org_alpha2results.items():
        for i in range(len(users_choices)):
            user_choices = users_choices[i]
            curr_item = history_dataset[i][-1]
            for j, item_index in enumerate(user_choices):  # delta loop
                item_index = int(item_index)
                reindexed_item = item_index

                T_tensor[i][curr_item, reindexed_item] += 1

                curr_item = reindexed_item

    for i in range(len(T_tensor)):
        T_tensor[i] = normalize_T(T_tensor[i], B)

        nodes, edges, weights = retrieve_parameters(
            T_tensor[i], reverse_videos_dict)

        filename = name + "_" + str(reverse_users_dict[i])
        path_file = graphs_folder + filename
        # print(path_file)
        create_graph_tsv(nodes, edges, weights, path_file, reverse_videos_labels_dict)

# Rec-guided Simulation
def generate_graphs(
        T_tensor,
        history_dataset,
        name,
        B,
        d,
        topk=10,
        num_items=0,
        num_users=100,
        users=None,
        reverse_users_dict=None,
        reverse_videos_dict=None,
        model_checkpoint_folder="",
        config=None,
        user_orientation_dict=None,
        item_label_dict=None,
        reverse_videos_labels_dict=None,
        graphs_folder="",
        args=None,
        dataset_path=None,
        c=1.0,
        gamma_list=[1.0, 1.0, 1.0],
        sigma_gamma_list=[1e-3, 1e-3, 1e-3],
        eta=0.01,
        introduce_bias=False,
        target="Horror",
        influence_percentage=0.0):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    warnings.simplefilter(
         'ignore',
         category=scipy.sparse.SparseEfficiencyWarning)

    print("NUM ITEMS", num_items)
    # hard cap for quick runs
    d = min(d, 15)
    B = min(B, 1)   # optional: also cap B so it only does 1 outer run

    SYNTHETIC = args["synthetic"]

    # --- NOVEL METRICS TOGGLE (safe default) ---
    # If args does not contain this flag, default to False (no metrics logging)
    log_metrics = False
    if args is not None:
        log_metrics = bool(args.get("log_metrics", False))


    model_users = np.array(users) + 1
    model_items = list(history_dataset.values())
    model_items = [np.array(y) + 1 for y in model_items]

    items_exposure = np.zeros((num_users, num_items))

    saving_histories = []

    for h in list(history_dataset.values()):
        saving_histories.append([reverse_videos_dict[x] for x in h])

    original_model, _, _, _, _, _ = load_model(
        model_checkpoint_folder=model_checkpoint_folder,
        config=config, num_items=num_items,
        args=args)

    if SYNTHETIC:
        X, Y, shrinkage_alpha, noise_cov = prepare_organic_model(dataset_path, reverse_videos_dict)
    else:
        organic_preferences = prepare_organic_model_movielens(history_dataset, item_label_dict)

    graphs_folder = graphs_folder + "{}_{}_gamma1_{}_sigmagamma1_{}_gamma2_{}_sigmagamma2_{}" \
                                    "_gamma3_{}_sigmagamma3_{}_eta_{}".format(c, 1-c, gamma_list[0], sigma_gamma_list[0],
                                                                               gamma_list[1], sigma_gamma_list[1],
                                                                               gamma_list[2], sigma_gamma_list[2], eta)

    if introduce_bias:
        graphs_folder += "_biased_{}_{}".format(target, influence_percentage)
        items_target_to_take = []
        all_items_target = []
        for item in range(num_items):
            if item_label_dict[item] == target:
                all_items_target.append(item)
        all_items_target = set(all_items_target)

        count = 0
        for u in range(len(history_dataset)):
            items_target_available = all_items_target - set(history_dataset[u])
            items_target_to_take.append(list(items_target_available))
            count += len(items_target_to_take[-1])
        print("Mean item {} potential".format(target), count / len(history_dataset))


    graphs_folder += "/"

    # --- NOVEL: enable diversity rerank + bubble metrics logging ---
    enable_diversity_rerank = False
    lambda_div = 0.5
    if args is not None:
        enable_diversity_rerank = bool(args.get("diversity_rerank", False))
        lambda_div = float(args.get("lambda_div", 0.5))

    metrics_path = os.path.join(graphs_folder, "bubble_metrics.csv")
    write_header = not os.path.exists(metrics_path)
# --------------------------------------------------------------

    
    a_list = []
    b_list = []

    for i in range(len(gamma_list)):
        if gamma_list[i] == 0 and sigma_gamma_list[i] == 0:
            a_value = -1
            b_value = -1
        else:
            a_value = gamma_list[i] * (gamma_list[i] * (1 - gamma_list[i]) / (sigma_gamma_list[i] ** 2) - 1)
            b_value = a_value * (1 / gamma_list[i] - 1)

        a_list.append(a_value)
        b_list.append(b_value)

    all_sessions = []

    for iter_b in range(B):

        start_time = time.time()

        print(iter_b, "iteration of B")

        temp_histories = copy.deepcopy(model_items)
        if introduce_bias:
            items_target_to_take_d = copy.deepcopy(items_target_to_take)

        sessions = []
        for s in list(temp_histories):
            sessions.append([reverse_videos_dict[x - 1] for x in s])

        temp_item_values = np.array([np.ones(len(x)) for x in temp_histories])

        interaction_dict = {
            "user_id": torch.LongTensor(model_users),
            "item_id": temp_histories,
            "item_value": temp_item_values
        }

        if iter_b == 0:
            model = copy.deepcopy(original_model)

        if SYNTHETIC:
            organic_preferences = generate_organic_preferences(X, Y, shrinkage_alpha, noise_cov)
        for j in range(d):
            #start_time_d = time.time()

            interactions = Interaction(interaction_dict)

            results = model.predict_for_graphs(interactions)

            scores = results.view(-1, num_items + 1).detach().cpu().numpy()

            scores[:, 0] = -np.inf  # set scores of [pad] to -inf

            nullify_history_scores(temp_histories, scores)

            scores = torch.FloatTensor(scores)

            topk_scores, topk_recommendations = torch.topk(scores, topk)

            temp_temp_item_values = []
            temp_temp_histories = []

            for i in range(len(topk_recommendations)):

                orientation_index = user_orientation_dict[i]
                user_recommendations = topk_recommendations[i].detach().cpu().numpy()

                if a_list[orientation_index] == -1 and b_list[orientation_index] == -1:
                    gamma = 0
                else:
                    gamma = np.random.beta(a_list[orientation_index], b_list[orientation_index], size=1)

                if np.random.binomial(1, gamma):
                    if np.random.binomial(1, eta):
                        items_available = list(set(np.arange(1, num_items + 1)) - set(temp_histories[i]))
                        item_sampled = np.random.choice(items_available, size=1)[0]

                    else:
                        organic_preferences_user_tmp = copy.deepcopy(organic_preferences[i])
                        organic_preferences_user_tmp[temp_histories[i] - 1] = -10000
                        item_sampled = np.argmax(organic_preferences_user_tmp) + 1  # +1 to be consistent
                else:
                    items_exposure[i][user_recommendations - 1] += 1

                    scores_probs = np.array(topk_scores[i])

                    temp = scores_probs / scores_probs.sum()

                    if np.any(temp) < 0:
                        scores_probs = softmax(temp)
                    else:
                        scores_probs = temp

                    if introduce_bias:

                        num_target_items_to_add = int(influence_percentage * topk)

                        if len(items_target_to_take_d[i]) < num_target_items_to_add:
                            items_target_to_add = items_target_to_take_d[i]
                        else:
                            items_target_to_add = np.random.choice(items_target_to_take_d[i], size=num_target_items_to_add,
                                                                   replace=False)

                        idx_to_add = 0
                        for idx, item_rec in enumerate(user_recommendations):
                            if item_label_dict[item_rec - 1] == target:
                                continue
                            user_recommendations[idx] = items_target_to_add[idx_to_add] + 1 # to be coherent with indexes
                            idx_to_add += 1
                            if idx_to_add >= len(items_target_to_add):
                                break


                    # Novel extension: diversity-aware reranking of the Top-K list (optional)
                    if enable_diversity_rerank and item_label_dict is not None:
                        user_recommendations, scores_probs = diversity_rerank_topk(
                            user_recommendations, scores_probs, item_label_dict, lambda_diversity=lambda_diversity
                        )

                    if c < 1:

                        organic_user_preferences = np.array(
                            [organic_preferences[i][int(l) - 1] for l in user_recommendations])

                        min_value = np.min(organic_user_preferences)
                        max_value = np.max(organic_user_preferences)
                        if min_value < 0.0 or max_value > 1.0:
                            organic_user_preferences = (organic_user_preferences - min_value) / (max_value - min_value)
                            organic_user_preferences[organic_user_preferences == 0.] = 0.001
                        if np.sum(organic_user_preferences) > 0:
                            organic_user_preferences /= np.sum(organic_user_preferences)

                            combination_probs = c * scores_probs + (1 - c) * organic_user_preferences
                        else:
                            combination_probs = scores_probs
                    else:
                        combination_probs = scores_probs

                    if np.min(combination_probs) < 0:
                        combination_probs = softmax(combination_probs)
                    else:
                        combination_probs /= np.sum(combination_probs)

                    item_sampled = np.random.choice(user_recommendations, p=combination_probs)


                    # Metrics logging (optional): category entropy of Top-K and of the user's history (a simple "filter-bubble" proxy)
                    if log_metrics and item_label_dict is not None:
                        topk_ent = category_entropy_from_items(user_recommendations - 1, item_label_dict)
                        hist_ent = category_entropy_from_items(np.array(temp_histories[i]) - 1, item_label_dict)
                        chosen_cat = item_label_dict.get(int(item_sampled - 1), "UNK")
                        with open(metrics_path, "a", encoding="utf-8") as mf:
                            mf.write(f"{iter_b},{j},{i},{topk_ent},{hist_ent},{int(item_sampled)},{chosen_cat}\n")

                if introduce_bias:
                    items_target_to_take_d[i] = list(set(items_target_to_take_d[i]) - {item_sampled - 1})

                T_tensor[i][temp_histories[i][-1] - 1, item_sampled - 1] += 1

                sessions[i].append(reverse_videos_dict[item_sampled - 1])

                temp_temp_histories.append(
                    np.append(temp_histories[i], item_sampled))
                temp_temp_item_values.append(
                    np.append(temp_item_values[i], 1.))

            temp_histories = np.array(temp_temp_histories)
            temp_item_values = np.array(temp_temp_item_values)

            interaction_dict["item_id"] = temp_histories
            interaction_dict["item_value"] = temp_item_values

            #print("Time for an iteration of d:", time.time() - start_time_d)

        all_sessions.append(sessions)
        print("Time for an iteration of B:", time.time() - start_time)

    for i in range(len(T_tensor)):
        T_tensor[i] = normalize_T(T_tensor[i], B)

        nodes, edges, weights = retrieve_parameters(
            T_tensor[i], reverse_videos_dict)

        filename = name + "_" + str(reverse_users_dict[i])

        path_file = graphs_folder + filename

        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)
        create_graph_tsv(nodes, edges, weights, path_file, reverse_videos_labels_dict)