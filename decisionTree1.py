import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import mpl_toolkits.mplot3d
from pympler import tracker
import json
import svmpy.svm as svm
import svmpy.Kernel as kernel


# from ranking_SVM.baseline import baseline
# from ranking_SVM.r_train import r_train
# from ranking_SVM.r_predict import r_predict


class Node(object):
    def __init__(self, data=-1, lchild=None, rchild=None, distribution=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild
        self.distribution = distribution


def label_fix(labelSet):
    selected_labels = np.array(labelSet)
    index = selected_labels[:, 0]
    labels = selected_labels[:, 1:]
    min_max_scaler = preprocessing.MinMaxScaler()
    labels = min_max_scaler.fit_transform(labels)
    selected_labels = np.vstack((index, labels.T)).T
    return selected_labels


# global
dataSet = pd.read_csv("../RF/features.csv", sep=',', header=-1)
rng = np.random.choice(np.arange(0, len(dataSet)), replace=False, size=len(dataSet))
trainSet = dataSet.iloc[rng[3:]]
testSet = dataSet.iloc[rng[0:3]]
labelSet = pd.read_csv("../RF/labels.csv", sep=',', header=-1)
labels = label_fix(labelSet)


def bootstrap(features, num):
    length = len(features)
    rng = np.random.choice(np.arange(0, length), replace=False, size=num)
    rng.sort()
    all_set = np.arange(length)
    all_set = list(all_set)
    out_of_bag_set = list(set(all_set).difference(set(rng)))
    bootstrap_set = features.iloc[rng]
    oob_set =  features.iloc[np.array(out_of_bag_set)]
    bootstrap_indexes = list(bootstrap_set[0])
    return bootstrap_set,oob_set


def features_sampling(features, num):
    weight = features.shape[1]
    length = features.shape[0]
    rng = np.random.choice(np.arange(1, weight), replace=False, size=num)
    rng.sort()
    cols = np.arange(0, weight, 1)
    selected_features = np.ones((num, length, weight))
    selected_index = rng
    for index in range(len(rng)):
        cols_list = list(cols)
        cols_list.insert(len(cols_list), cols_list.pop(cols_list[int(rng[index])]))
        selected_features[index, :, :] = features.ix[:, cols_list]
    return selected_features, selected_index


def tree_split(features, selected_index):
    num = features.shape[0]
    row = features.shape[1]
    col = features.shape[2]

    split_points = np.zeros(num)
    split_information_values = np.zeros(num)
    split_indexes = np.zeros(num)
    child_set_index = []
    for x in range(num):
        sorted_features = features[x, features[x, :, col - 1].argsort(), :]
        effect_feature = sorted_features[:, col - 1]
        split_value = 0.5 * (effect_feature[1:] + effect_feature[:-1])
        features_index = sorted_features[:, 0]
        child_set_index.append(features_index)
        selected_labels = labels[np.argwhere([x == labels[:, 0] for x in features_index])[:, 1]]
        #   observe_matrix = selected_labels[:, 1:]
        split_index, split_information = split_score_function(selected_labels)
        split_point = split_value[split_index]
        split_indexes[x] = split_index
        split_points[x] = split_point
        split_information_values[x] = split_information
        # label_vector(selected_labels)
        ##labelization
        # m = np.ones((int(row),int(row)-1))
        # label_matrix = np.triu(m)
        # label_split_value=sorted_features[:,col-1]
        # print(123)
        # for icol in range(label_matrix.shape[1]):
        #     y = label_matrix[:,icol]
        #     x = sorted_features[:,0:col-1]
        #     clf=tree.DecisionTreeClassifier()
        #     list.insert(clfs,len(clfs),clf)
        #     clf.fit(x,y)
        # clf = clf.fit(iris.data, iris.target)
    split_feature_index = split_information_values.argmin()
    split_point = split_points[split_feature_index]
    split_feature = selected_index[split_feature_index]
    child_index = child_set_index[split_feature_index]
    split_index = split_indexes[split_feature_index]

    return split_point, split_feature, split_feature_index, split_index, child_index


def KL_divergence(p, q):
    # normalize
    # min_max_scaler = preprocessing.MinMaxScaler()
    # p = min_max_scaler.fit_transform(p)
    # q = min_max_scaler.fit_transform(q)
    p = p + 0.000001
    q = q + 0.000001
    value = (-p * np.log(p / q)).sum()
    return value


def Shannon(p, q, m, n):
    k = (m / (m + n)) * p + (n / (m + n)) * q
    value = (m / (m + n)) * KL_divergence(p, k) + (n / (m + n)) * KL_divergence(q, k)
    return value


def Shannon2(p, q, m, n):
    k = 0.5 * (q + p)
    value = (m / (m + n)) * KL_divergence(p, k) + (n / (m + n)) * KL_divergence(q, k)
    return value


def error_sum(p, q):
    value = (np.absolute(p - q)).sum()
    return value


def Jensen_Shannon_divergence(p, q):
    k = 0.5 * (q + p)
    value = 0.5 * KL_divergence(p, k) + 0.5 * KL_divergence(q, k)
    return value


def split_score_function(selected_labels):
    KL_value_a = np.zeros(selected_labels.shape[0])
    KL_inverse_value = np.zeros(selected_labels.shape[0])
    JSD = np.zeros(selected_labels.shape[0])
    Shannon_value = np.zeros(selected_labels.shape[0])
    errors = np.zeros(selected_labels.shape[0])
    class_a_set = np.zeros((selected_labels.shape[0], selected_labels.shape[1]))
    class_b_set = np.zeros((selected_labels.shape[0], selected_labels.shape[1]))
    # plt.figure(figsize=(100, 1), dpi=70)
    for x in range(1, selected_labels.shape[0]):
        class_a_split = selected_labels[:x, :]
        class_b_split = selected_labels[x:, :]
        # class b averaging
        class_b = np.mean(class_b_split, axis=0)
        class_a = np.mean(class_a_split, axis=0)
        # JSD[x] = Jensen_Shannon_divergence(class_a[1:], class_b[1:])
        # Shannon_value[x] = Shannon2(class_a[1:], class_b[1:], class_a_split.shape[0], class_b_split.shape[0])
        Shannon_value[x] = Shannon(class_a[1:],class_b[1:],class_a_split.shape[0],class_b_split.shape[0])
        # errors[x] = error_sum(class_a[1:], class_b[1:])
        # KL_value_a[x] = KL_divergence(class_a[1:], class_b[1:])


        # test & observe
    #     class_a_set[x] = class_a
    #     class_b_set[x] = class_b
    #     errors[x] = error_sum(class_a[1:], class_b[1:])
    #     KL_value_a[x] = KL_divergence(class_a[1:], class_b[1:])
    #     KL_inverse_value[x] = KL_divergence(class_b[1:], class_a[1:])
    #     JSD[x] = Jensen_Shannon_divergence(class_a[1:], class_b[1:])
    #     Shannon_value[x] = Shannon(class_a[1:],class_b[1:],class_a_split.shape[0],class_b_split.shape[0])
    #     p1 = plt.subplot(1, selected_labels.shape[0], x)
    #     p1.plot(np.array(class_a[1:]), c='blue')
    #     p1.plot(np.array(class_b[1:]), c='red')
    #     p1.set_title(x)
    #     p1.legend()
    #
    # plt.show()
    min_index = Shannon_value[1:].argmin()
    min = Shannon_value[1:].min()
    return min_index, min


def label_cluster(selected_labels):
    vectors = selected_labels[:, 1:]
    vectors = pd.DataFrame(vectors)
    kmeans_model_2 = KMeans(n_clusters=2, max_iter=1000)
    kmeans_model_2.fit(vectors)
    tsne = TSNE(learning_rate=500, n_components=2)
    tsne.fit_transform(vectors)

    oridata = pd.DataFrame(vectors, index=vectors.index)
    data = pd.DataFrame(tsne.embedding_, index=vectors.index)
    order2data = pd.DataFrame(oridata, index=oridata.index)
    c21 = data[kmeans_model_2.labels_ == 0]
    c22 = data[kmeans_model_2.labels_ == 1]
    # c23=data[kmeans_model_2.labels_==2]
    plt.plot(c21[0], c21[1], 'r.')
    plt.plot(c22[0], c22[1], 'go')
    plt.show()
    # 3 dimension
    # fig=plt.figure()
    # fig.set_size_inches(30,30)
    # fig.savefig('test2png.png', dpi=100)
    # ax=fig.add_subplot(111,projection='3d')
    # #ax.scatter(d21[0],d21[1],d21[2],c='g',marker=u'.')
    # ax.scatter(c21[0],c21[1],c21[2],c='r')
    # ax.scatter(c22[0],c22[1],c22[2],c='g')
    # fig.show()
    # # plt.show()
    # c2d=order2data[usr_stat_4['label']=='1.0']#有标签且标签是1的样本


def forest_gen(tree_num,bootstrap_num,feature_per_tree):
    forest = []
    oob_set=set('')
    for i in range(tree_num):
        random_samples,oob = bootstrap(trainSet, bootstrap_num)
        used_features = []
        tree_root = Node([-1, 0, 0])
        tree = Node()
        tree_gen(random_samples, used_features, tree, tree_root,feature_per_tree)
        # train tree
        for x in np.array(random_samples):# trainSet or random_samples
            tree_search(tree, x)
        # oob test
        # OOB_error(oob,tree)
        forest.append(tree)
        # oob_set|=set(bootstrap_indexes)
        # tr.print_diff()
        # print(i)
    # print("build complete")

    return forest


def tree_gen(random_samples, used_features, node, father_node, feature_per_tree):
    features, selected_index = features_sampling(random_samples, feature_per_tree)
    split_point, \
    split_feature, \
    split_feature_index, \
    split_index, \
    child_index = tree_split(features, selected_index)
    node.data = [split_feature, split_point, len(random_samples)]

    # if (used_features.count(split_feature) != 0):
    #     print(str(split_feature)+"-feature is used")
    while (father_node.data[0] != node.data[0] and (
                    node.data[2] != 2 and (node.lchild is None and node.rchild is None))):

        used_features.append(split_feature)
        random_samples = np.array(random_samples)

        left_child = random_samples[
            np.argwhere([x == random_samples[:, 0] for x in child_index[:int(split_index) + 1]])[:, 1]]
        right_child = random_samples[
            np.argwhere([x == random_samples[:, 0] for x in child_index[1 + int(split_index):]])[:, 1]]

        if (len(left_child[:, 0]) > 1):
            lnode = Node()
            node.lchild = lnode
            left_child = pd.DataFrame(left_child)
            tree_gen(left_child, used_features, lnode, node,feature_per_tree)

        if (len(right_child[:, 0]) > 1):
            rnode = Node()
            node.rchild = rnode
            right_child = pd.DataFrame(right_child)
            tree_gen(right_child, used_features, rnode, node,feature_per_tree)

    return 0


def tree_gen1(random_samples, used_features, node):
    features, selected_index = features_sampling(random_samples, 10)
    split_point, \
    split_feature, \
    split_feature_index, \
    split_index, \
    child_index = tree_split(features, selected_index)
    node.data = [split_feature, split_point, len(random_samples)]

    # if (used_features.count(split_feature) != 0):
    #     print(str(split_feature)+"-feature is used")
    while (used_features.count(split_feature) == 0):
        used_features.append(split_feature)
        random_samples = np.array(random_samples)
        left_child = random_samples[
            np.argwhere([x == random_samples[:, 0] for x in child_index[:int(split_index) + 1]])[:, 1]]
        right_child = random_samples[
            np.argwhere([x == random_samples[:, 0] for x in child_index[1 + int(split_index):]])[:, 1]]

        if (len(left_child[:, 0]) > 1):
            lnode = Node()
            node.lchild = lnode
            left_child = pd.DataFrame(left_child)
            tree_gen1(left_child, used_features, lnode)
        if (len(right_child[:, 0]) > 1):
            rnode = Node()
            node.rchild = rnode
            right_child = pd.DataFrame(right_child)
            tree_gen1(right_child, used_features, rnode)
    return 0


def property_fuction():
    return 0


def tree_search(node, features):
    feature_index = None
    split = None
    feature_score = None
    if (node.data[0] > -1):
        feature_index = node.data[0]
    if (node.data[1] > 0):
        split = node.data[1]
        feature_score = features[feature_index]

        if (feature_score < split):
            if (node.lchild != None):
                if ((node.lchild).data[0] != -5):
                    tree_search(node.lchild, features)
                else:
                    (node.lchild).data[1].append(features[0])
            else:
                index_set = [features[0]]
                NNode = Node([-5, index_set])
                node.lchild = NNode
        if (feature_score > split):
            if (node.rchild != None):
                if ((node.rchild).data[0] != -5):
                    tree_search(node.rchild, features)
                else:
                    (node.rchild).data[1].append(features[0])
            else:
                index_set = [features[0]]
                NNode = Node([-5, index_set])
                node.rchild = NNode
    return 0


def leaf_distribution(node):
    feature_index = None
    split = None
    indexes = None
    if (node.data[0] != -5):
        if (node.lchild != None):
            leaf_distribution(node.lchild)

        if (node.rchild != None):
            leaf_distribution(node.rchild)
    if (node.data[0] == -5):
        indexes = node.data[1]
        distribution = distribution_matrix(indexes)
        node.distribution = distribution

    return 0


def leaf_search(node, features):
    feature_index = None
    split = None
    feature_score = None
    similar_index_set = None
    if (node.data[0] > -1):
        feature_index = node.data[0]
    if (node.data[1] > 0):
        split = node.data[1]
        feature_score = features[feature_index]

        if (feature_score < split):
            if (node.lchild != None):
                if ((node.lchild).data[0] != -5):
                    similar_index_set = leaf_search(node.lchild, features)
                else:
                    similar_index_set = (node.lchild).data[1]

        if (feature_score > split):
            if (node.rchild != None):
                if ((node.rchild).data[0] != -5):
                    similar_index_set = leaf_search(node.rchild, features)
                else:
                    similar_index_set = (node.rchild).data[1]

    return similar_index_set


def distribution_matrix(indexes):
    a = np.zeros((5, labelSet.shape[1] - 1))
    b = np.zeros((5, labelSet.shape[1] - 1))
    for x in indexes:
        label = np.zeros(labelSet.iloc[0].shape)
        try:
            label = labelSet.iloc[np.argwhere(labelSet[0] == x)[0]]
        except:
            print(x, indexes)
        label = np.array(label)
        label = label.astype('int')
        label = label.astype('str')

        for k, v in np.ndenumerate(label[0][1:]):
            b[int(v) - 1, k] = 1
        a += b
    return a


def train_one(forest, test_index):
    test = trainSet.iloc[test_index - 1]

    for i in range(len(forest)):
        tree_search(forest[i], test)
    return 0


def train_many(forest):
    for i in range(len(forest)):
        for x in np.array(trainSet):
            tree_search(forest[i], x)
    # print("train complete..")
    # leaf distribution compute
    # for i in range(len(forest)):
    #     leaf_distribution(forest[i])
    return forest


def count(data):
    count_frq = dict()
    for one in data:
        if one in count_frq:
            count_frq[one] += 1
        else:
            count_frq[one] = 1
    count_frq_v = sorted(count_frq.items(), key=lambda d: d[1], reverse=True)
    count_frq_k = sorted(count_frq.items(), key=lambda d: d[0], reverse=True)

    return count_frq, count_frq_v


def semantic_neighbor(forest, test):
    sn_set = []
    for i in range(len(forest)):

        sn = leaf_search(forest[i], test)

        if(sn!=None):
            sn.sort()
            sn_set.extend(sn)
    snn_map, count_frq_v = count(sn_set)
    return snn_map, count_frq_v

def OOB_error(oob_set,tree):
    model = [tree]
    true_err_set = []
    for test in np.array(oob_set):
        sn, count_frq_v = semantic_neighbor(model, test)
        tag = []
        label_array = np.array(labelSet)

        chosen = label_array[np.argwhere(label_array[:, 0] == count_frq_v[0][0])][0][0]
        test_label = label_array[np.argwhere(label_array[:, 0] == test[0])][0][0]
        error_tab = []
        expect_tab = []
        for row in label_array:
            err = error_sum(row[1:], test_label[1:])
            error_tab.append((row[0], err))
        error = pd.DataFrame(error_tab)
        for row in label_array:
            err = error_sum(row[1:], chosen[1:])
            expect_tab.append((row[0], err))
        expect = pd.DataFrame(expect_tab)

        max_times = 0
        target_members = []
        for index, v in enumerate(count_frq_v):
            if (v[1] > max_times):
                max_times = v[1]
            if (v[1] == max_times):
                target_members.append(v[0])
        target_list = []
        for n in target_members:
            target_list.append(label_array[np.argwhere(label_array[:, 0] == n)][0][0])
        target_tab = pd.DataFrame(target_list)
        true_err = error_sum(chosen[1:], test_label[1:])
        # print("123")
        true_err_set.append(true_err)
        # pd.DataFrame(dataSet.iloc[np.array([chosen, test_label])])
        # pd.DataFrame(labelSet.iloc[np.array([chosen, test_label])])

    err = np.mean(true_err_set)
    return err


def predict(model):
    K = 50
    true_err_set = []
    for test in np.array(testSet):
        sn, count_frq_v = semantic_neighbor(model, test)
        tag = []
        label_array = np.array(labelSet)

        # T format
        for k in count_frq_v:
            tag.append(label_array[np.argwhere(label_array[:, 0] == k[0])][0][0])
        tag = np.array(tag)
        tag_indexes = tag[:, 0]
        # T = np.zeros(label_array.shape)
        # T[:, 0] = label_array[:, 0]
        # for index, element in np.ndenumerate(T[:, 0]):
        #     if ((element == tag_indexes).any()):
        #         T[index] = tag[np.argwhere(tag[:, 0] == element)][0][0]
        # T[:, 1:][T[:, 1:] > 0] = 1
        # Z = T[:, 1:].sum()

        tag[:, 1:][tag[:, 1:] > 0] = 1
        T = tag
        count_frq_v = count_frq_v[:K]  # nearest K neighbors
        T = T[:K]  # nearest K neighbors

        # D format
        # Ntree = len(model)
        # D = np.zeros((K, Ntree))
        # for index, v in enumerate(count_frq_v):
        #     D[index, int(v[1])] = 1
        # D_t = pd.DataFrame(D)
        # psi = np.dot(T[:, 1:].T, D)
        # print("123")
        chosen = label_array[np.argwhere(label_array[:, 0] == count_frq_v[0][0])][0][0]
        test_label = label_array[np.argwhere(label_array[:, 0] == test[0])][0][0]
        error_tab = []
        expect_tab = []
        for row in label_array:
            err = error_sum(row[1:], test_label[1:])
            error_tab.append((row[0], err))
        error = pd.DataFrame(error_tab)
        for row in label_array:
            err = error_sum(row[1:], chosen[1:])
            expect_tab.append((row[0], err))
        expect = pd.DataFrame(expect_tab)

        max_times = 0
        target_members = []
        for index, v in enumerate(count_frq_v):
            if (v[1] > max_times):
                max_times = v[1]
            if (v[1] == max_times):
                target_members.append(v[0])
        target_list = []
        for n in target_members:
            target_list.append(label_array[np.argwhere(label_array[:, 0] == n)][0][0])
        target_tab = pd.DataFrame(target_list)
        true_err = error_sum(chosen[1:], test_label[1:])
        print("123")
        true_err_set.append(true_err)
    err = np.mean(true_err_set)
    return err


# def model_save(model):
#     for i in range(len(model)):
#         model[i]
#
# def node2json(Node):
#     if(Node.data!=0)


if __name__ == '__main__':
    s = 0
    times = 20
    treeNum = 100
    bootstrapNum = 225
    testNum = 3
    feature_per_tree = 10 #108
    for i in range(times):
        # update
        rng = np.random.choice(np.arange(0, len(dataSet)), replace=False, size=len(dataSet))
        trainSet = dataSet.iloc[rng[testNum:]]
        testSet = dataSet.iloc[rng[0:testNum]]
        # model
        model = forest_gen(treeNum,bootstrapNum,feature_per_tree)
        # train_many(model)
        p =predict(model)
        print("err"+str(p))
        s+=p

    print("All "+str(s))
    print("mean "+str(s/times))
    print("tree "+str(treeNum))
    print("bootstrapNum "+str(bootstrapNum))
    print("feature_per_tree "+str(feature_per_tree))

