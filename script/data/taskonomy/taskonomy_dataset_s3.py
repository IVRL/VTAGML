import os
import io
import boto3
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO: Fix these images and then remove this


from .task_configs import task_parameters
from .transforms import task_transform
from .splits import get_splits


filter_amount_dict = {'low': 10000, 'medium': 50000, 'high': 100000}
forbidden_buildings = [
    'mosquito', 'tansboro', 'tomkins', 'darnestown', 'brinnon', # We do not have the rgb data for tomkins, darnestown, brinnon
    'rough', 'grace', 'wiconisco' # Contain some wrong viewpoints
]


class TaskonomyDatasetS3(Dataset):
    def __init__(self,
                 tasks,
                 split='train',
                 variant='fullplus',
                 rm_incomplete=True,
                 image_size=256,
                 max_images=None,
                 seed=0,
                 filter_amount='medium'):
        '''
        Taskonomy EPFL-S3 dataloader.
        Make sure the environment variables S3_ENDPOINT, S3_TASKONOMY_ACCESS,
        S3_TASKONOMY_KEY, and S3_TASKONOMY_BUCKET are set.

        Args:
            tasks: List of tasks
            split: One of {'train', 'val', 'test', 'all'}
            variant: One of {'debug', 'tiny', 'medium', 'full', 'fullplus'}
            rm_incomplete: Set to True to only keep samples that have every task
            image_size: Target image size
            max_images: Optional subset selection
            seed: Random seed for deterministic shuffling order
            filter_amount: How many "bad" images to remove. One of {'low', 'medium', 'high'}.
        '''
        super(TaskonomyDatasetS3, self).__init__()
        self.tasks = tasks
        self.split = split
        self.variant = variant
        self.rm_incomplete = rm_incomplete
        self.image_size=image_size
        self.max_images = max_images
        self.seed = seed
        self.filter_amount = filter_amount

        # S3 bucket setup
        self.session = boto3.session.Session()
        self.s3_client = self.session.client(
            service_name='s3',
            aws_access_key_id=os.environ.get('S3_TASKONOMY_ACCESS'),
            aws_secret_access_key=os.environ.get('S3_TASKONOMY_KEY'),
            endpoint_url=os.environ.get('S3_ENDPOINT')
        )
        self.bucket_name = os.environ.get('S3_TASKONOMY_BUCKET')

        #  DataFrame containing information whether or not any file for any task exists
        self.df_meta = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'metadata', 'taskonomy_files.pkl.gz'))

        # Select splits based on selected size/variant
        splits = get_splits(
            os.path.join(os.path.dirname(__file__), 'metadata', f'train_val_test_{variant}.csv'),
            forbidden_buildings=forbidden_buildings
        )
        if split == 'all':
            self.buildings = list(set(splits['train']) | set(splits['val']) | set(splits['test']))
        else:
            self.buildings = splits[split]
        self.buildings = sorted(self.buildings)
        self.df_meta = self.df_meta.loc[self.buildings]

        # Filter bad images
        df_filter = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'metadata', 'taskonomy_filter_scores.pkl.gz'))
        df_filter = df_filter[:filter_amount_dict[filter_amount]]
        filtered_indices = self.df_meta.index.difference(df_filter.index)
        self.df_meta = self.df_meta.loc[filtered_indices]

        self.df_meta = self.df_meta[tasks] # Select tasks of interest
        if rm_incomplete:
            # Only select rows where we have all the tasks
            self.df_meta = self.df_meta[self.df_meta.all(axis=1)]
        self.df_meta = self.df_meta.sample(frac=1, random_state=seed) # Random shuffle
        self.df_meta = self.df_meta[:max_images] if max_images is not None else self.df_meta # Select subset if so desired

        print(f'Using {len(self.df_meta)} images from variant {self.variant} in split {self.split}.')


    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, index):

        # building / point / view are encoded in dataframe index
        building, point, view = building, point, view = self.df_meta.iloc[index].name
        # TODO: Remove this try/except after we made sure there are no bad/missing images!
        # Very slow if it fails.
        try:

            result = {}
            for task in self.tasks:
                # Load from S3 bucket
                ext = task_parameters[task]['ext']
                domain_id = task_parameters[task]['domain_id']
                key = f'taskonomy_imgs/{task}/{building}/point_{point}_view_{view}_domain_{domain_id}.{ext}'
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)['Body'].read()

                # Convert bytes to image / json / array / etc...
                if ext == 'png':
                    file = Image.open(io.BytesIO(obj))
                elif ext == 'json':
                    file = json.load(io.BytesIO(obj))
                    if task == 'point_info':
                        file['building'] = building
                        file.pop('nonfixated_points_in_view')
                elif ext == 'npy':
                    file = np.frombuffer(obj)
                else:
                    raise NotImplementedError(f'Loading extension {ext} not yet implemented')

                # Perform transformations
                file = task_transform(file, task=task, image_size=self.image_size)

                result[task] = file

            return torch.stack([result[self.tasks[0]],result[self.tasks[0]]]), torch.stack([result[t].view(-1,self.image_size,self.image_size) for i,t in enumerate(self.tasks) if i!=0] ),torch.LongTensor([i for i in range(len(self.tasks)-1)])

        
        except Exception as e :
            # In case image was faulty or not uploaded yet, try with random other image

            return self[np.random.randint(len(self))]
