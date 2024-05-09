from config import config

def main_operation_loop():
    print('[INFO] selecting operation')

    if config.operation == 'train_3d_segmentation':
        print('[INFO] performing 3d segmentation training...')
        from train_operations.train_3d_seg import perform_training
        perform_training()

    elif config.operation == 'train_2d_segmentation':
        print('[INFO] performing 2d segmentation training...')
        from train_operations.train_2d_seg import perform_training
        perform_training()
    
    elif config.operation == 'test_3d_segmentation':
        print('[INFO] performing 3d segmentation testing')
        from train_operations.inference_3d_seg import perform_inferencing
        perform_inferencing()
    elif config.operation == 'test_2d_segmentation':
        print('[INFO] performing 2d segmentation testing')
        from train_operations.inference_2d_seg import perform_inferencing
        perform_inferencing()

    elif config.operation == 'inference_testing':
        print('[INFO] performing 2d segmentation testing')
        from train_operations.inference_2d_seg import inference_testing
        inference_testing()
    else:
        print('[INFO] no valid operation was selected')
