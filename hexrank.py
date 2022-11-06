import numpy as np
import json

class Hex:
    # basic configuration
    dist_levels = [10, 30, 60]
    travel_modes = ["foot", "bike", "car"]

    def __init__(self, name):
        self.name = name
        hexes = self.parse_hexagons()
        self.n = len(hexes)
        print(self.n)
        self.raw_features = self.read_feature_matrix()
        self.hex_centers = np.asarray([h["center"] for h in hexes])
        self.feature_dim = self.raw_features.shape[-1]

        # precompute/read distance matrices
        self.distance_matrices = {}
        for travel_mode in self.travel_modes:
            self.distance_matrices[travel_mode] = self.read_dist_matrix(travel_mode)

        # precompute all hex ranks 
        feature_request = {
        "penalty": "logistic",
        }
        self.hex_ranks = {}
        for travel_mode in self.travel_modes:
            for distance_level in self.dist_levels:
                feature_request["dists"] = travel_mode
                feature_request["half_value_time"] = distance_level
                self.hex_ranks[travel_mode + "-" + str(distance_level)] = self.create_features(feature_request)

    def parse_hexagons(self):
        with open("./data/" + self.name + "_hexagons.json", "r") as f:
            hex_data = json.load(f)["features"]
            hexes = []
            for hex in hex_data:
                coords = hex["geometry"]["coordinates"][0]
                entry = {
                    "center": np.mean(np.asarray(coords[0:6]), axis=0),
                    "coords": np.asarray(coords[0:6]) 
                }
                hexes.append(entry)
            return hexes

    def read_feature_matrix(self):
        with open("./data/"+self.name+"_features.txt") as f:
            F = np.sqrt(np.asarray([[float(num) for num in line.split(" ")] for line in f]))
            return F

    def read_dist_matrix(self, travel_mode):
        with open("./data/"+self.name+"_"+str(travel_mode)+"_dists.txt", "r") as f:
            D = np.asarray([[float(num) for num in line.split(" ")] for line in f]) / 60.0
            return D

    def compute_test_dists(self):
        X2 = np.repeat(np.sum(self.hex_centers**2, axis=1), self.n).reshape(self.n, self.n)
        XXT = self.hex_centers @ self.hex_centers.T
        dists = np.sqrt(np.maximum(np.zeros((self.n, self.n)), X2 + X2.T - 2 * XXT))
        return dists

    def linear_dist_transform(self, dist_mat):
        max_val = np.max(dist_mat)
        # max distance locations have zero value now
        return 1 - dist_mat / max_val

    def bell_dist_transform(self, dist_mat, a):
        return np.exp(-(dist_mat**2) / (2 * a**2))

    def create_features(self, request_details):
        # transform distance matrix entries
        if request_details["dists"] not in self.distance_matrices:
            print("Distance metric not suported")
            return
        dmat = self.distance_matrices[request_details["dists"]]
        dist_penalty = request_details["penalty"]
        if (dist_penalty == "linear"):
            dmat = self.linear_dist_transform(dmat)
            hexranks = dmat @ self.raw_features
        elif (dist_penalty == "logistic"):
            hvt = request_details["half_value_time"]
            tr_dmat = self.bell_dist_transform(dmat, hvt)
            hexranks = tr_dmat @ self.raw_features
        
        # scale every feature component
        feature_maxs = np.max(hexranks, axis=0)
        for i in range(self.feature_dim):
            if feature_maxs[i] == 0.0:
                feature_maxs[i] = 1.0
        hexranks = np.divide(hexranks, np.repeat(feature_maxs, self.n).reshape((self.feature_dim, self.n)).T)
        return hexranks

    def request_features(self, request_details, save_path = "./output/hexrank"):
        # we have precomputed everything (hopefully)
        hexranks = np.zeros((self.n, self.feature_dim))
        for i in range(self.feature_dim):
            dist = request_details["dist_measure"][i]
            dist_lvl = request_details["dist_level"][i]
            hexranks[:,i] = self.hex_ranks[dist+"-"+str(self.dist_levels[dist_lvl])][:, i]

        # TODO: maybe apply weights here 

        if save_path is not None:
            self.save_features(hexranks, save_path)
        
        return hexranks

    def save_features(self, features, save_path):
        hexrank_list = {}
        hexrank_list["type"] = "Hexranks"
        hexrank_list["features"] = features.tolist()
        json_features = json.dumps(hexrank_list)
        with open(save_path + ".json", "w") as outfile:
            outfile.write(json_features)



if __name__ == "__main__":
    testHex = Hex("helsinki")

    for transport_mode in ["foot", "bike", "car"]:
        for dist_level in [0,1,2]:
            hex_request = {
                "dist_measure" : [transport_mode]*6,
                "dist_level" : [dist_level] * 6
                }
            testHex.request_features(hex_request, "./output/helsinki_"+transport_mode+"_"+str(dist_level))