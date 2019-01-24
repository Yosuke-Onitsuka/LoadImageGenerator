class LoadImageGenerator(Sequence):
    def __init__(self, x_paths, y, batch_size):
        self.x_paths = x_paths
        self.y = y
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.x_paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x_paths = self.x_paths[idx*self.batch_size : min(len(self.x_paths), (idx+1)*self.batch_size)]
        batch_y = self.y[idx*self.batch_size : min(len(self.y), (idx+1)*self.batch_size)]
        
        image = [self.horizontal_flip(self.vertical_flip(img_to_array(load_img(path)))) for path in batch_x_paths] 
        
        return np.array(image), np.array(batch_y)
    
    def horizontal_flip(self, image, rate=0.5):
        if np.random.rand() < rate:
            image = image[:, ::-1, :]
        return image
    
    def vertical_flip(self, image, rate=0.5):
        if np.random.rand() < rate:
            image = image[::-1, :, :]
        return image
