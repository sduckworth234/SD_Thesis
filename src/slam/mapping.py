# mapping.py

import numpy as np

class Map:
    def __init__(self):
        self.map_data = {}

    def add_landmark(self, landmark_id, position):
        self.map_data[landmark_id] = position

    def get_landmark(self, landmark_id):
        return self.map_data.get(landmark_id, None)

    def remove_landmark(self, landmark_id):
        if landmark_id in self.map_data:
            del self.map_data[landmark_id]

    def get_all_landmarks(self):
        return self.map_data.items()

    def clear_map(self):
        self.map_data.clear()

def main():
    # Example usage
    mapping = Map()
    mapping.add_landmark('1', np.array([1.0, 2.0, 3.0]))
    print(mapping.get_all_landmarks())

if __name__ == "__main__":
    main()