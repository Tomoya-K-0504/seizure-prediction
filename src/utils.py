
class_names = ['interictal', 'preictal']
subject_dir_names = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']


def arrange_paths(args, sub_name):
    args.train_manifest = 'splitted/{}/train_manifest.csv'.format(sub_name)
    args.val_manifest = 'splitted/{}/val_manifest.csv'.format(sub_name)
    args.test_manifest = 'splitted/{}/test_manifest.csv'.format(sub_name)
    args.model_path = 'model/' + args.model_name + '/{}.pkl'.format(sub_name)
    args.log_id = sub_name

    return args