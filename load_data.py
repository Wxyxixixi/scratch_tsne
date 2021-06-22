from sklearn import datasets


def dataLoader():
    print("Start loading....")
    examples = datasets.load_digits()
    data = examples.data
    label = examples.target
    print("Loading successfully")

    return data, label
