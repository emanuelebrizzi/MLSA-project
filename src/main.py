from load_dataset import load_codesearchnet
from preprocess import preprocess_dataset

def main():
    dataset = load_codesearchnet()

    dataset = preprocess_dataset(dataset)

    print(dataset)

if __name__ == "__main__":
    main()