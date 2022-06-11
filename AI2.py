import math
import copy
import matplotlib.pyplot as plt
import time


def nearestNeighbor(data_points, data_point, subset_of_features, num_instances):
    # Stores the nearest neighbor
    nn = 0
    # initializing it to high value, later will store nearest neighbor
    shortest_distance = 999999

    for instance in range(num_instances):
        if data_point == instance:
            pass
        # find nearest neighbor and classify the point
        else:
            distance = 0
            # Get distance, compare and update nearestNeighbor
            for j in range(len(subset_of_features)):
                distance = distance + pow((data_points[instance][subset_of_features[j]] - data_points[data_point][subset_of_features[j]]), 2)
            distance = math.sqrt(distance)

            if distance < shortest_distance:
                nn = instance
                shortest_distance = distance
    return data_points[nn][0]


def kCross_validation(data_points, subset_of_features, number_of_instances):
    # As explained in the report it performs K-fold cross validation where k=1
    # This checks if the test datapoint is equal to class of nearest neighbor
    right_classfication = 0.0
    for i in range(number_of_instances):
        oneDatapoint = i
        neighbor = nearestNeighbor(data_points, oneDatapoint, subset_of_features, number_of_instances)
        if neighbor == data_points[oneDatapoint][0]:
            right_classfication = right_classfication + 1

    accuracy = (right_classfication / number_of_instances) * 100
    return accuracy

def plotGraph(finalResults,algo_option):
    # creating the dataset

    features = list(finalResults.keys())
    accuracy = list(finalResults.values())

    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(features, accuracy, color='green',
            width=0.4)
    plt.title("Accuracy v/s Feature set")
    plt.ylabel("Accuracy")
    if algo_option==1:
        plt.xlabel("Forward Selection")
    elif algo_option==2:
        plt.xlabel("Backward Elimination")
    # To adjust the x-axis label values
    plt.xticks(rotation=30)
    plt.show()


# Function to perform forward selection
def forwardSelection(data_points, number_of_instances, number_of_features,algo_option):
    start_time=time.time()
    # This variable is later used to plot the graph
    output={}
    subset_of_features = []
    output_set = []
    # Variable to store the highest accuracy
    final_accuracy = 0.0

    for i in range(number_of_features):
        add_feature,temp_add = 0,0
        temp_accuracy = 0.0
        for j in range(1, number_of_features + 1):
            if j not in subset_of_features:
                # If feature j is not in the subset then go ahead. Else ignore
                # We do not want to change the original set , so we are creating a deepcopy and changing it here
                temp_subset = copy.deepcopy(subset_of_features)
                temp_subset.append(j)
                accuracy = kCross_validation(data_points, temp_subset, number_of_instances)

                print('\t Feature/s ', temp_subset, ' has accuracy of ', accuracy, '%')
                if accuracy > final_accuracy:
                    final_accuracy,add_feature = accuracy,j
                if accuracy > temp_accuracy:
                    temp_accuracy,temp_add = accuracy,j

        if add_feature != 0:
            subset_of_features.append(add_feature)
            output_set.append(add_feature)
            output[str(subset_of_features)]=final_accuracy
            print('\n\nFeature set ', subset_of_features, ' had the best accuracy of ', final_accuracy, '%\n\n')
        else:
            print('\n\n(Decreased accuracy. Will keep searching if local maxima is found)')
            subset_of_features.append(temp_add)
            print('Feature set ', subset_of_features, ' had the best accuracy of ', temp_accuracy, '%\n\n')
            output[str(subset_of_features)] = final_accuracy

    print('Search complete \n The best feature subset is', output_set, ' with accuracy of: ', final_accuracy, '%')
    end_time = time.time()
    total_time = abs(start_time-end_time)
    print("Time taken to find the best feature:", round(total_time),'seconds')
    # Call function to plot bar graph for the computed values
    plotGraph(output,algo_option)

# Function to perform backward elimination
def backwardElimination(data_points, number_of_instances, number_of_features, final_acc,algo_option):
    start_time = time.time()
    # It works similar to forward selection. But we start with a set of all features and eliminate one by one.
    output = {}
    # Start with full feature set
    subset_of_features = [i + 1 for i in range(number_of_features)]
    output_set = [i + 1 for i in range(number_of_features)]
    # Set current accuracy to accuracy found before feature algorithm
    final_accuracy = final_acc
    for i in range(number_of_features):
        remove_feature,local_remove = 0,0

        temp_accuracy = 0.0
        for j in range(1, number_of_features + 1):

            if j in subset_of_features:
                temp_subset = copy.deepcopy(subset_of_features)
                temp_subset.remove(j)

                accuracy = kCross_validation(data_points, temp_subset, number_of_instances)
                print('\t Feature/s ', temp_subset, ' has accuracy of ', accuracy, '%')
                if accuracy > final_accuracy:
                    final_accuracy = accuracy
                    remove_feature = j
                if accuracy > temp_accuracy:
                    temp_accuracy = accuracy
                    local_remove = j
        if remove_feature != 0:
            subset_of_features.remove(remove_feature)
            output_set.remove(remove_feature)
            output[str(subset_of_features)] = final_accuracy
            print('\n\nFeature set ', subset_of_features, ' had the best accuracy of ', final_accuracy, '%\n\n')
        else:
            print('\n\n(Decreased accuracy. Will keep searching if local maxima is found)')
            subset_of_features.remove(local_remove)
            output[str(subset_of_features)] = final_accuracy
            print('Feature set ', subset_of_features, ' had the best accuracy of ', temp_accuracy, '%\n\n')

    print('Search complete \n The best feature subset is', output_set, ' with accuracy of: ', final_accuracy, '%')
    end_time = time.time()
    total_time=abs(start_time-end_time)
    print("Time taken to find the best feature:", round(total_time),'seconds')
    # Call function to plot bar graph for the computed values
    plotGraph(output,algo_option)

def main():
    print('Nearest neighbor algorithm with feature selection')
    print('Select the type of dataset to run the algorithm')
    print('1. Small dataset')
    print('2. Large dataset')
    dataset_choice = int(input())

    if dataset_choice == 1:
        file = "/Users/ankithas/Desktop/AI2/CS205_SP_2022_SMALLtestdata__32.txt"
    elif dataset_choice == 2:
        file = "/Users/ankithas/Desktop/AI2/CS205_SP_2022_Largetestdata__72.txt"
    else:
        print('Invalid choice!! Choose option 1 or 2')

    # Open the given file store in data and
    # In case of failure , return error.
    try:
        data = open(file, 'r')
    except:
        raise IOError('Entered file does not exist')

    # Read in first line to see # features
    lineOne = data.readline()

    # Number features = total in one line - 1, -1 because first column tells which class it belongs to
    numberOf_features = len(lineOne.split()) - 1

    # Reset cursor to first line
    data.seek(0)
    # Number of instances = Total number of lines in the file
    numberOf_instances = sum(1 for line in data)
    data.seek(0)

    # Store data in an 2D array
    instances = [[] for i in range(numberOf_instances)]
    for i in range(numberOf_instances):
        instances[i] = [float(j) for j in data.readline().split()]

    print('Number of features:', numberOf_features)
    print('Number of instances:', numberOf_instances)

    # Ask user to provide their choice of algorithm
    print('Choose the search algorithm to run:')
    print('1. Forward Selection')
    print('2. Backward Elimination\n')
    algo_option = int(input())

    # Normalize data before classification
    print('Normalizing dataset.....')
    min_value, max_value = [], []
    for i in range(1, numberOf_features + 1):
        min_value.append(min(row[i] for row in instances))
        max_value.append(max(row[i] for row in instances))

    # Applying Min-Max normalization
    # Feature x (scaled) = (x - min(x) / max(x) - min(x) ) * 10
    for i in range(0, numberOf_instances):
        for j in range(1, numberOf_features + 1):
            instances[i][j] = ((instances[i][j] - min_value[j - 1]) / (max_value[j - 1] - min_value[j - 1])) * 10
    # Normalized data is stored in below variable
    normalized_data = instances
    print('Normalization done')
    # Run nearest neighbor + one out validation + ALL features, print results
    feature_list = []
    for i in range(1, numberOf_features + 1):
        feature_list.append(i)

    # Performing K-Cross validation
    accuracy = kCross_validation(normalized_data, feature_list, numberOf_instances)
    print('Running nearest neighbor with all ', numberOf_features,'features, with K fold cross validation where K=1')
    print('Accuracy achieved in ',accuracy,'%\n')

    print('Beginning search\n')

    if algo_option == 1:
        forwardSelection(normalized_data, numberOf_instances, numberOf_features,algo_option)
    elif algo_option == 2:
        backwardElimination(normalized_data, numberOf_instances, numberOf_features, accuracy,algo_option)
    else:
        print('Invalid choice!! Choose 1 or 2.')


if __name__ == '__main__':
    main()
