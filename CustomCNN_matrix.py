import numpy as np

# BasicNet
conf_matrix = np.array([
    [122, 2, 4, 22, 0, 0, 0],
    [5, 117, 5, 1, 7, 4, 11],
    [6, 1, 136, 2, 1, 3, 1],
    [18, 0, 0, 120, 1, 7, 4],
    [5, 2, 3, 1, 135, 0, 4],
    [7, 11, 4, 7, 3, 118, 0],
    [5, 5, 13, 9, 2, 2, 114]
])

# ImprovedNet
# conf_matrix = np.array([
#     [96, 7, 0, 28, 9, 2, 8],
#     [7, 85, 5, 1, 18, 2, 32],
#     [13, 7, 96, 11, 5, 2, 16],
#     [30, 2, 7, 89, 2, 4, 16],
#     [6, 5, 3, 5, 127, 0, 4],
#     [3, 8, 4, 14, 3, 111, 7],
#     [7, 5, 10, 0, 8, 1, 119]
# ])

# ImprovedNetLite
# conf_matrix = np.array([
#     [119, 3, 4, 13, 11, 0, 0],
#     [5, 102, 7, 4, 8, 5, 19],
#     [5, 1, 140, 3, 0, 1, 0],
#     [24, 0, 3, 116, 2, 0, 5],
#     [1, 3, 0, 0, 138, 0, 8],
#     [4, 8, 8, 8, 3, 113, 6],
#     [0, 4, 8, 3, 6, 0, 129]
# ])


# Class names
class_names = [
    'Audi', 'Hyundai_Creta', 'Mahindra_Scorpio',
    'Rolls_Royce', 'Swift', 'Tata_Safari', 'Toyota_Innova'
]

# Calculate metrics
num_classes = len(class_names)
metrics = {}

for i in range(num_classes):
    tp = conf_matrix[i, i]
    fp = np.sum(conf_matrix[:, i]) - tp
    fn = np.sum(conf_matrix[i, :]) - tp
    tn = np.sum(conf_matrix) - tp - fp - fn

    accuracy = (tp + tn) / np.sum(conf_matrix)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics[class_names[i]] = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }


# Print metrics
for class_name, metric_values in metrics.items():
    print("Metrics for class:", class_name)
    for metric_name, value in metric_values.items():
        print(metric_name + ":", value)
    print()
