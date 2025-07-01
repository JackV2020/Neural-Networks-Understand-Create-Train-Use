---

# Improved scripts

After finishing the story on neural networks I decided to improve the training and put that in a seperate directory.

The reason why I did not include all these improvements in the first place is that I wanted to keep the script as simple as possible and explain what it is doing in detail without to much ovehead.

So what is changed here?

The script script_generate_pictures_improved.py creates 3 seperate folders with pictures.

 - training: contains the data used during the training sessions
 - validation: contains the data used during validation
 - test: contains data which you can use to test and is very much like the data in training and validation 
    (these should all report (close to) 100% certainty)

Not a big deal, just did it because I liked to split it up.

This is what it is about.

The script script_train_model_improved.py has extra functionality:

 - training will stop early to use less time and avoid overfitting when 
    - validation loss gets worse 5 times in a row (the best result is saved)
    - validation reaches a level we think is good enough
    - it might decide to stop after 3 of the 20 epochs
 - the training inludes a little augmentation to improve the quality of the training
    - it rotates the image between -5 and +5 degrees for 70% of the examples
    - it adds some noise for 20% of the samples
    - it blurs 20% of the samples 
 - the training script produces a keras model and a tflite model
 - the validation shows a progress bar just like the training
 - AND BEST : load data per batch
    - this allows you to train very big datasets as long as training + validation batches fit in memory
    
Comments from the script_train_model.py were replaced by new comments to explain the new functionality 

You can use the test scripts from the code folder to test the models created by this script_train_model_improved.py.

---

## Notes on training and validation results:

During training you see long lines containing training results:

 - color_accuracy: 0.9586   : 95.86% predections were oke   
 - color_loss: 0.1072       : 'how wrong were the errors'
 - loss: 3.0656             : = color_loss + perc_loss
 - perc_accuracy: 0.2832    : 28.32% predections were oke   
 - perc_loss: 2.9584        : 'how wrong were the errors'

Each training batch aims to minimize the loss, which indirectly improves accuracy.

Both accuracy and loss are outcomes of the predictions, but only the loss is used to update the model weights during training.

Smaller batches lead to more updates per epoch, potentially improving generalization, but may also introduce more noise.

And after validation completes you see extra results:

 - val_color_accuracy: 1.0000   : 100% predections were oke    
 - val_color_loss: 0.0033       : 'how wrong are the errors' 
 - val_loss: 0.0448             : = val_color_loss + val_perc_loss
 - val_perc_accuracy: 0.9983    : 99.83% predections were oke    
 - val_perc_loss: 0.0415        : 'how wrong were the errors'
 
These validation results are just info and not used for the training.

Training accuracy that increases more slowly than validation accuracy with augmentation is OK and often a good sign. It shows that the model is learning more robustly.

---

## So what's next

Next could be:

 - use more or less training and validation data
 - now we have batch loading of the images use bigger pictures
 - change parameters in the training script
 - add more augmentation options
 - check validation results and model performance
 - use a black line to draw a basic shape (rectangle, triangle,.. ) and add shape for third set of output neurons
    - draw.rectangle([x_left, y_top, x_right, y_bottom], outline=None, width=5)
    - draw.polygon([(x1, y1), (x2, y2), (x3, y3)], outline=None, width=5)
    - draw.ellipse([x_left, y_top, x_right, y_bottom], outline=None, width=5)
    - create directory names like blue_21_ellipse, red_42_triangle and green_84_square
    - use augmentation to rotate things between -180 and +180 degrees
    (may be a percentage killer so maybe add self.augment_train_above_50 for percentages above 50%?)
 - send me any comments you have

Thanks for reading this. If you made it through without panicking, well done — there truly was no reason to panic, and as you can see, it was mostly harmless all along.

Jack
