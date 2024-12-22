if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.
    """
    (x_train, _), _ = load_dataset("mnist")
    centers, errors = lloyd_algorithm(x_train, 10)
    
    for image in centers:
        image = image.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
    


if __name__ == "__main__":
    main()
