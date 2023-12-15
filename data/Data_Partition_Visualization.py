import matplotlib.pyplot as plt

if __file__ == "__main__":

    # Given JSON data
    data = {
        "num_clients": 20,
        "num_classes": 10,
        "non_iid": True,
        "balance": False,
        "partition": "pat",
        "Size of samples for labels in clients": [
            [[0, 1233], [1, 1101]],
            [[0, 407], [1, 911]],
            [[0, 1268], [1, 1865]],
            [[0, 3995], [1, 4000]],
            [[2, 1021], [3, 307]],
            [[2, 1134], [3, 927]],
            [[2, 318], [3, 686]],
            [[2, 4517], [3, 5221]],
            [[4, 1584], [5, 1457]],
            [[4, 1475], [5, 1435]],
            [[4, 1372], [5, 514]],
            [[4, 2393], [5, 2907]],
            [[6, 1085], [7, 434]],
            [[6, 639], [7, 1696]],
            [[6, 1078], [7, 850]],
            [[6, 4074], [7, 4313]],
            [[8, 568], [9, 1412]],
            [[8, 732], [9, 1174]],
            [[8, 750], [9, 926]],
            [[8, 4775], [9, 3446]]
        ],
        "alpha": 0.1,
        "batch_size": 10
    }

    # Extract label size distribution for each client from the given data
    client_data = data["Size of samples for labels in clients"]

    # Create a bar chart
    plt.figure(figsize=(10, 5))  # Set the figure size

    for i, client in enumerate(client_data):
        x = [label[0] for label in client]  # Label index as x-axis
        y = [label[1] for label in client]  # Sample size as y-axis
        plt.bar(x, y, label=f"Client {i + 1}")  # Plot the bar chart for each client

    # Set the labels and title of the chart
    plt.xlabel("Label")
    plt.ylabel("Sample Size")
    plt.title("Data Partition Visualization")

    # Add a legend
    plt.legend()

    # Show the chart
    plt.show()
