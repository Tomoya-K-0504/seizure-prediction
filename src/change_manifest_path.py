from pathlib import Path
import pandas as pd


if __name__ == '__main__':

    manifest_path = Path('/media/cs11/C420894C20894700/koike/archive/splitted/Dog_3/test_manifest.csv')
    target_dir = '/media/cs11/C420894C20894700/koike/archive/splitted/'
    orig_df = pd.read_csv(manifest_path, header=None)

    new_df = orig_df.applymap(lambda x: target_dir + '/'.join(x.split('/')[-3:]))
    new_df.to_csv(manifest_path, index=False, header=None)