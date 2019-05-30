import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


def parsing_one_xml(xml_path, label):
    features, labels = [], []
    tree = ET.parse(xml_path)
    fragments = tree.findall('*/WordFragment')
    for fragment in fragments:
        features.append([
            int(fragment.attrib['BlackCount']),
            int(fragment.attrib['WhiteHolesCount']),
            int(fragment.attrib['HorzStrokesCount']),
            int(fragment.attrib['VertStrokesCount']),
            int(fragment.attrib['MaxHorzStrokeLength'])
        ])
        labels.append(label)
    return features, labels


def parsing_xml(path, label, features_mask=None):
    features, labels = [], []
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            xml_path = os.path.join(path, filename)
            tree = ET.parse(xml_path)
            fragments = tree.findall('*/WordFragment')
            for fragment in fragments:
                features.append([
                    int(fragment.attrib['BlackCount']),
                    int(fragment.attrib['WhiteHolesCount']),
                    int(fragment.attrib['HorzStrokesCount']),
                    int(fragment.attrib['VertStrokesCount']),
                    int(fragment.attrib['MaxHorzStrokeLength'])
                ])
                labels.append(label)
    return features, labels


text_path, nontext_path = 'E:\\test_task\\Text', 'E:\\test_task\\Nontext'
text_label, nontext_label = 1, 0
text_data, text_labels = parsing_xml(text_path, text_label)
nontext_data, nontext_labels = parsing_xml(nontext_path, nontext_label)
X, y = text_data + nontext_data, text_labels + nontext_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

clf = RandomForestClassifier(criterion='entropy', max_depth=50, max_features='auto',
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0, n_estimators=50)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)  # probabilities (n_samples, n_classes)
precision, recall, _ = precision_recall_curve(y_test, probs[:, 1], pos_label=1)
fpr, tpr, _ = roc_curve(y_test, probs[:, 1], pos_label=1)

plt.plot(recall, precision)
plt.grid(b=True, axis='both', color='.9')
plt.title('PR curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(fpr, tpr)
plt.grid(b=True, axis='both', color='.9')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print(metrics.auc(fpr, tpr))
print(metrics.auc(recall, precision))

classes = ['nontext', 'text']
print(classification_report(y_test, clf.predict(X_test), target_names=classes))

# parameter, accuracy = [], []
# for i in range(0, 100):
#     clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#                                  max_depth=50, max_features='auto', max_leaf_nodes=None,
#                                  min_impurity_decrease=0.0, min_impurity_split=None,
#                                  min_samples_leaf=1, min_samples_split=2,
#                                  min_weight_fraction_leaf=0, n_estimators=50, n_jobs=None,
#                                  oob_score=False, random_state=None, verbose=0,
#                                  warm_start=False)
#     clf.fit(X_train, y_train)
#     parameter.append(i)
#     accuracy.append(clf.score(X_test, y_test))
#     print(i)
# plt.plot(parameter, accuracy)
# plt.show()
#
# clf = RandomForestClassifier()
# param_grid2 = {"n_estimators": [35, 40],
#                "max_depth": [40, 50, 60],
#                "min_samples_split": [2, 3, 4],
#                "min_samples_leaf": [1, 2],
#                "min_weight_fraction_leaf": [0, 0.01, 0.02]}
# grid_search = GridSearchCV(clf, param_grid=param_grid2, cv=5)
# grid_search.fit(X_train, y_train)
# best_clf = grid_search.best_estimator_
# print(best_clf)
# print(best_clf.score(X_test, y_test))
