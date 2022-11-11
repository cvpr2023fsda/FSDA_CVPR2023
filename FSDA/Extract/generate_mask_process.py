import numpy as np
from sklearn.cluster import KMeans

def fourier_intensity_extraction(x_spectrum, idxx, idxy, num_clusters, image_size) :
    x_spectrum = x_spectrum.cpu().detach().numpy()
    # plot_plor_coordinate(x_spectrum[0])
    data = []
    for idxx_, idxy_ in zip(idxx, idxy):
        data.append([idxx_, idxy_, np.sum(x_spectrum[:,  idxx_, idxy_])])

    num_pix = np.zeros((len(data), 1))
    sum_int = np.zeros((len(data), 1))

    for i in range(len(data)):
        num_pix[i] = len(data[i][0])
        sum_int[i] = data[i][2]

    X = sum_int / (num_pix + 1)
    km = KMeans(n_clusters=num_clusters, random_state=4321)
    km.fit(X)
    labels = km.predict(X)

    data_np = np.array(data)
    clustering_data = []

    for label in np.unique(labels):
        clustering_data.append(data_np[np.where(labels == label)[0].tolist()])

    clustered_idx = []
    for label in np.unique(labels):
        mask = np.zeros((image_size, image_size))
        for label_ in clustering_data[label] :
            idxx, idxy = label_[0], label_[1]
            mask[idxx, idxy] = 1
        clustered_idx.append(np.where(mask == 1))

    return clustered_idx, X, labels

def get_small_region(image_size, angle, length, preserve_range):
    idxx, idxy = [], []
    start = preserve_range
    x_range = np.arange(0, image_size) - int(image_size / 2)
    y_range = np.arange(0, image_size) - int(image_size / 2)

    x_ms, y_ms = np.meshgrid(x_range, y_range)

    R = np.sqrt(x_ms ** 2 + y_ms ** 2)
    T = np.degrees(np.arctan2(y_ms, x_ms))

    T[T < 0] += 360

    # length = 2 * np.random.choice([1, 2, 3, 4, 5], 1, False)[0]
    # angle = 2 * np.random.choice([1, 2, 3, 4, 5], 1, False)[0]

    for l in range(start, image_size // 2, length):
        for d in range(0, 360, angle):
            idxx_, idxy_ = get_small_region_idx(image_size, angle, length, R, T, l, d)
            idxx.append(idxx_); idxy.append(idxy_)

    return idxx, idxy

def get_small_region_idx(image_size, angle, length, R, T, l, d):
    if l + length <= image_size // 2 : idxx, idxy = np.where((R > l) & (R <= l + length) & (T >= d) & (T < d + angle))
    else : idxx, idxy = np.where((R > l) & (R <= image_size//2) & (T >= d) & (T < d + angle))

    return idxx, idxy

def plot_plor_coordinate(spectrum) :
    import matplotlib.pyplot as plt
    import sys
    theta = np.linspace(0, 2 * np.pi, 1000)
    length = np.linspace(0, int(np.sqrt(2 * 64**2)), 1000)
    for l in range(0, int(np.sqrt(2 * 64**2)), 2) :
        x1 = l * np.cos(theta) + spectrum.shape[0] // 2
        x2 = l * np.sin(theta) + spectrum.shape[1] // 2
        plt.plot(x1, x2, 'k-')

    for a in np.radians(range(0, 360, 2)) :
        x1 = length * np.cos(a) + spectrum.shape[0] // 2
        x2 = length * np.sin(a) + spectrum.shape[1] // 2
        plt.plot(x1, x2, 'k-')

    plt.imshow(np.log(1 + spectrum), cmap='inferno')
    plt.axis('off'); plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    # plt.show()
    plt.savefig('small_region.png', bbox_inches='tight', pad_inches=0, dpi=300)
    sys.exit()