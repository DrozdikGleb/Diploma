import numpy as np
from sklearn.tree import DecisionTreeClassifier

from feature_extraction.MetaFeatureApi import MetaFeature
from sklearn.tree._tree import Tree


class DecisionTreeMeta(MetaFeature):
    def getLength(self) -> int:
        height = 1
        leaves_num = 1
        node_num = 1
        width = 1
        dev = 4
        max = 4
        mean = 4
        min = 3
        result = sum([height, leaves_num, node_num, width, dev, max, mean, min])
        return result

    @staticmethod
    def getHeight(tree: Tree) -> int:
        children_left = tree.children_left
        children_right = tree.children_right

        def walk(node_id):
            if children_left[node_id] != children_right[node_id]:
                left_max = 1 + walk(children_left[node_id])
                right_max = 1 + walk(children_right[node_id])
                return max(left_max, right_max)
            else:  # leaf
                return 1

        root_node_id = 0
        return walk(root_node_id)

    @staticmethod
    def getLeavesNumber(tree: Tree) -> int:
        children_left = tree.children_left
        children_right = tree.children_right

        def walk(node_id):
            if children_left[node_id] != children_right[node_id]:
                left_max = 0 + walk(children_left[node_id])
                right_max = 0 + walk(children_right[node_id])
                return left_max + right_max
            else:  # leaf
                return 1

        root_node_id = 0
        return walk(root_node_id)

    @staticmethod
    def getNodeNumber(tree: Tree) -> int:
        children_left = tree.children_left
        children_right = tree.children_right

        def walk(node_id):
            if children_left[node_id] != children_right[node_id]:
                left_max = 0 + walk(children_left[node_id])
                right_max = 0 + walk(children_right[node_id])
                return 1 + left_max + right_max
            else:  # leaf
                return 0

        root_node_id = 0
        return walk(root_node_id)

    @staticmethod
    def getWidth(tree: Tree, height: int) -> int:
        children_left = tree.children_left
        children_right = tree.children_right

        def walk(node_id, level):
            if level == 0:
                return 1
            if children_left[node_id] == children_right[node_id]:
                return 0
            if children_left[node_id] != children_right[node_id]:
                left_max = walk(children_left[node_id], level - 1)
                right_max = walk(children_right[node_id], level - 1)
                return left_max + right_max
            else:  # leaf
                return 0

        root_node_id = 0
        width = 0
        for i in range(height):
            width = max(width, walk(root_node_id, i))
        return width

    def getAttrs(self, tree: Tree) -> [float]:
        attrs = [0.0] * self.features
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature

        def walk(node_id, counts) -> [float]:
            if children_left[node_id] != children_right[node_id]:
                counts[feature[node_id]] += 1
                left = walk(children_left[node_id], counts)
                right = walk(children_right[node_id], left)
                return right
            else:
                return counts

        root_node_id = 0
        return walk(root_node_id, attrs)

    def getBranches(self, tree: Tree) -> [float]:
        branches = [0.0] * self.getLeavesNumber(tree)
        children_left = tree.children_left
        children_right = tree.children_right

        def walk(node_id, counts, lenght, index) -> [float]:
            if children_left[node_id] == children_right[node_id]:
                counts[index] += lenght
                return index + 1
            else:
                left = walk(children_left[node_id], counts, lenght + 1, index)
                right = walk(children_right[node_id], counts, lenght + 1, left)
                return right

        root_node_id = 0
        walk(root_node_id, branches, 0, 0)
        return branches

    def getClasses(self, tree: Tree) -> [float]:
        classes = [0.0] * 2
        children_left = tree.children_left
        children_right = tree.children_right
        value = tree.value

        def walk(node_id, counts) -> [float]:
            if children_left[node_id] == children_right[node_id]:
                class_no = np.argmax(value[node_id][0])
                counts[class_no] += 1
                return counts
            else:
                left = walk(children_left[node_id], counts)
                right = walk(children_right[node_id], left)
                return right

        root_node_id = 0
        return walk(root_node_id, classes)

    def getLevels(self, tree: Tree) -> [float]:
        height = self.getHeight(tree)
        levels = [0.0] * (height + 1)
        children_left = tree.children_left
        children_right = tree.children_right

        def walk(node_id, counts, level, height) -> [float]:
            if children_left[node_id] == children_right[node_id]:
                return
            if height == 0:
                counts[level] += 1
            walk(children_left[node_id], counts, level, height - 1)
            walk(children_right[node_id], counts, level, height - 1)
            return

        root_node_id = 0
        for i in range(height):
            walk(root_node_id, levels, i, i)
        return levels

    def max_(self, inp: [float]) -> float:
        return max(inp)

    def mean_(self, inp: [float]) -> float:
        return np.mean(np.array(inp)).item(0)

    def min_(self, inp: [float]) -> float:
        return min(inp)

    def dev_(self, inp: [float]) -> float:
        return np.std(np.array(inp)).item(0)

    def getMax(self, getAttrs, getBranches, getClasses, getLevels) -> [float]:
        attr = self.max_(getAttrs)
        branches = self.max_(getBranches)
        classes = self.max_(getClasses)
        levels = self.max_(getLevels)
        return [attr, branches, classes, levels]

    def getMin(self, getAttrs, getBranches, getClasses) -> [float]:
        attr = self.min_(getAttrs)
        branches = self.min_(getBranches)
        classes = self.min_(getClasses)
        return [attr, branches, classes]

    def getDev(self, getAttrs, getBranches, getClasses, getLevels) -> [float]:
        attr = self.dev_(getAttrs)
        branches = self.dev_(getBranches)
        classes = self.dev_(getClasses)
        levels = self.dev_(getLevels)
        return [attr, branches, classes, levels]

    def getMean(self, getAttrs, getBranches, getClasses, getLevels) -> [float]:
        attr = self.mean_(getAttrs)
        branches = self.mean_(getBranches)
        classes = self.mean_(getClasses)
        levels = self.mean_(getLevels)
        return [attr, branches, classes, levels]

    def getMeta(self, zero_in: np.ndarray, one_in: np.ndarray) -> np.ndarray:
        data_in = self.data(zero_in, one_in)
        labels_in = self.labels()
        d_tree = DecisionTreeClassifier(random_state=0)
        d_tree.fit(data_in, labels_in)
        tree = d_tree.tree_

        height = self.getHeight(tree)
        leaves_num = self.getLeavesNumber(tree)
        node_num = self.getNodeNumber(tree)
        width = self.getWidth(tree, height)

        attr = self.getAttrs(tree)
        branches = self.getBranches(tree)
        classes = self.getClasses(tree)
        levels = self.getLevels(tree)

        max = self.getMax(attr, branches, classes, levels)
        min = self.getMin(attr, branches, classes)
        dev = self.getDev(attr, branches, classes, levels)
        mean = self.getMean(attr, branches, classes, levels)
        result = [height, leaves_num, node_num, width]
        result.extend(max)
        result.extend(min)
        result.extend(dev)
        result.extend(mean)
        return np.array(result)


if __name__ == '__main__':
    meta = DecisionTreeMeta(4, 3)
    arr = np.array([[1, 1, 1, 1],
                    [0.2, 0.3, 0, 0],
                    [0.2, 0, 0, 1]])
    print(meta.getMeta(arr, arr - 0.5))
    print(meta.getLength())
