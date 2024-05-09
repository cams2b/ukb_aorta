

class config:

    experiment_name = ''
    model_name = ''
    output_path = ''

    train_path = ''
    validation_path = ''

    test_path = ''

    operation = '' # train_2d_segmentation, test_2d_segmentation


    epochs = 25
    batch_size = 64
    
    patch_size = [240, 240, 1]
    use_slice = True
    patches_per_image = 40
    label_probabilities = {0 : 0.5, 1 : 0.5}
    queue_length = 200

    learning_rate = 0.001
    lr_patience = 5
    resume_training = False
    weight_path = ''


    ### inferencing operations
    generate_volume =True
    generate_metrics = True
    save_prediciton = True
    save_all = True






