[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## License

This project is licensed under the terms of the Unlicense.
<br>For more details, please refer to [UNLICENSE.md](UNLICENSE.md).
<br>For more information, please refer to <http://unlicense.org>.

---

# Neural Networks

I wrote this story to explain neural networks as I understand them in an easy understandable way and it ends with a working command line application and a working web application.

The command line tool analyzes simple pictures. The web application allows you to upload pictures and shows the analysis results on the web page.
When you run the web application on a mobile it allows you to take a picture and upload that for analysis.

Everything you need to create and train the neural network and run both applications is in the **code directory** so you do not need to create these yourself. 

<i>Ignore the **code_improved directory** until you survived the **code directory**</i>

I did my utmost to write operating system independent code so I trust everything runs on Windows too.

Today is a hot sunny day in June 2025, and I have a bright and shiny state‑of‑the‑art laptop from 2005 (only 20 years young!) with an i7 processor and 8 GB of RAM, running Lubuntu on a modest 255 GB SSD, yes the disk was replaced once it was affordable.

That’s what I’m using to write this an used to get the network up and running with python.

While official temperatures rise above 35 °C and everyone stays in a cool environment, I start with some very simple math.

"Don’t panic" the neural network will do the math for us in the end, but first we need to understand how to tell it what to do.

During the explanation you may even think, “hmmm, that’s handy”.

So first, we look at the math. Then we dive into a very simple example of what we could use a neural network for and how it uses the math.

After this we have a theoretical basic neural network with just one single element a **neuron** and understand how it works.

Then we look at a python script of 5 lines to demonstrate how we can implement the complete math for the neural network.

After that we create a script to create a very simple picture in a file and another script to read the picture file and feed that to the neural network to do the math.

Next we’ll expand the idea to use more neurons and discuss a more advanced network.

By then, you’ll have the basic knowledge to assemble a real neural network and realize that building one from scratch is an immense job. You might even feel it's impossible without a Guide.

We could could continue without, but modest as we are, we accept the help to avoid that we throw ourselves at the ground and miss...

So with some help we create, train and use a real neural network on my bright and shiny laptop and also create a command line tool and a web application to use the neural network.

---

## Some math made simple

Before jumping into neural networks, I’ll explain some math in a simple way, using basic examples so I understand it myself.

Let’s imagine we have a grocery store and we only sell 3 products: apples, pears, and potatoes.

To keep it easy, we sell fixed amounts and use prices chosen for easy calculation:

- Apples: € 1 for 2 kg
- Pears: € 2 for 1 kg
- Potatoes: € 3 for 5 kg

We can put this in a structured table which we call a 3 x 2 matrix:

(3 x 2 matrix because we have 3 rows and 2 columns)

```
         weight  price
apples     2       1
pears      1       2
potatoes   5       3
```

Customers can buy any combination. For example:

Person 1 buys 2×2 kg apples, 1×5 kg potatoes (no pears)
Person 2 buys 3×1 kg pears, 1×5 kg potatoes (no apples)
Person 3 buys 1×2 kg apples, 2×1 kg pears, 2×5 kg potatoes

We represent this in another table, a 3 x 3 matrix:

```
         apples  pears  potatoes
Person 1   2       0       1
Person 2   0       3       1
Person 3   1       2       2
```

Note: we include zeros for missing items to form a nice rectangular table.

Using these two tables, we compute for each person:

- **Total weight** = the sum of every : count × weight_per_item
- **Total price** =  the sum of every : count × price_per_item

Example calculations:

- **Person 1**
  Weight = 2×2 + 0×1 + 1×5 = 9
  Price  = 2×1 + 0×2 + 1×3 = 5

- **Person 2**
  Weight = 0×2 + 3×1 + 1×5 = 8
  Price  = 0×1 + 3×2 + 1×3 = 9

- **Person 3**
  Weight = 1×2 + 2×1 + 2×5 = 9
  Price  = 1×1 + 2×2 + 2×3 = 5

The results can be put in another 3 x 2 matrix :

```
         weight  price
Person 1    9       5
Person 2    8       9
Person 3    9       5
```

These tables are called **matrices**, and the process to calculate the last matrix is **matrix multiplication**.

And this is a complicated matrix. Make sure you understand this because we are going to simplify....

---

### Simplifying for neural networks

To simplify, let’s focus on Person 1 and create vectors:

A vector is a notation between square brackets.

Row vector (purchases): `[2, 0, 1]`
Column vector (weights):
```
    [2
     1
     5]
```

Matrix Multiplication for this single row and column:

(^T below means values were on top of each other like above but are now in a row):
(^T officially : transposed)
```
[2, 0, 1] × [2, 1, 5]^T = 2 × 2 + 0 × 1 + 1 × 5 = 9
or in general
[p0, p1, p3, ... ] x [w0, w1, w2, ...]^T = p0 x w0 + p1 x w1 + p2 x w2 + ... = result

```

This is the key mechanism used in neural networks, though there will typically be many more values than 3.

You can skip to the next section or read the 'Bonus'.

```
Bonus:

When you want to multiply 2 matrices :
  the number of columns of the first matrix must match 
  the number of rows in the second matrix.

You can check that with the calculation we made in our grocery above.

In general When you multiply two matrices :  P x W = R 
for                  matrix P       x         matrix W
you calculate (columns P x rows P ) x ( columns W x rows W)
which results in size    (columns P x rows W)
result                   (columns R x rows R)
result                          matrix R

So for a multiplication of a 1 x 3 with a 3 x 1 matrix like before we end up with a 1 x 1 matrix.

And for a 4 x 1 and 1 x 4 you also get a 1 x 1 matrix which is the answer to our question:

[2, 7, 3, 2] x [5, 2, 4, 3] = 10 + 14 + 12 + 6 = 42
```

---

### Our neural network example

Imagine a 'high‑definition' :smiley: camera that captures a 4×4 = 16 pixel black and white picture.

We want to detect how many of the 4 central pixels are white (e.g. indicating something is in the center).

Label pixels `p0` to `p15`, and convert black to 0, white to 1.

Arrange them:

```
p0  p1  p2  p3
p4  p5  p6  p7
p8  p9  p10 p11
p12 p13 p14 p15
```

Our input picture with a center white is the base for `w0` to `w15`:

```
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0
```

The neural network we build looks like the schema below.

```
p0  ----->   i0----[w0  = 0]-------------------+
p1  ----->   i1----[w1  = 0]-----------------+ |
p2  ----->   i2----[w2  = 0]---------------+ | |
p3  ----->   i3----[w3  = 0]-------------+ | | |
p4  ----->   i4----[w4  = 0]-----------+ | | | |
p5  ----->   i5----[w5  = 1]---------+ | | | | |
p6  ----->   i6----[w6  = 1]-------v v v v v v v
p7  ----->   i7----[w7  = 0]---->[1 single output]----->    result
p8  ----->   i8----[w8  = 0]-----^ ^ ^ ^ ^ ^ ^ ^
p9  ----->   i9----[w9  = 1]-------+ | | | | | |
p10 ----->   i10---[w10 = 1]---------+ | | | | |
p11 ----->   i11---[w11 = 0]-----------+ | | | |
p12 ----->   i12---[w12 = 0]-------------+ | | |
p13 ----->   i13---[w13 = 0]---------------+ | |
p14 ----->   i14---[w14 = 0]-----------------+ |
p15 ----->   i15---[w15 = 0]-------------------+

```

Our 16 pixels p0..p15 are the input for the 16 inputs i0..i15.

All inputs are multiplied by a weight and totaled by '1 singel output' to get the result of white pixels in the center.

This means: result = p0 x w0 + ... + p15 x w15

The part in the center is a basic neuron with 16 inputs and 1 single output

Here, the result is 4 when the central 4 pixels are white,  2 if shifted up and 1 when in a corner.

This is our 'single-neuron network'.

---

### Normalizing output between 0 and 1

Neural networks often scale outputs between 0 and 1.

We achieve this by reducing the weights: each of the 1’s becomes `0.25`, zeros stay zero.

Now:

- When all 4 center pixels are white: result = 1
- When 2 are white: result = 0.5
- In a corner: 0.25

Remember this 'tuning the weights to get what we want' as training the neural network which we will do later.

---

## Python code for single neuron neural network

I needed a long title for this section since the code is only 5 lines and 3 lines do the work.

We are about to create a python script with numpy to do the matrix multiplication.

numpy is short for Numerical Python which supports matrix calculations.

In the beginning I mentioned that it is a script of 5 lines.

We can discuss this length for a "long time" because there are some empty lines, the first line is borrowing something, then 2 lines we need to have something to do and at last a line which only shows off what we achieved.

Anuway, create a script with the next in it and save it with the name matrix.py:

```python
import numpy as np

P = np.array([1,1,0,0, 1,1,0,0,       0,0,0,0,       0,0,0,0]) # 2x2 in top left corner
W = np.array([0,0,0,0, 0,0.25,0.25,0, 0,0.25,0.25,0, 0,0,0,0]) # the weights (after our training)

R = np.dot(P, W) # matrix multiplication is a dot product

print(f"Percentage white: {round(R * 100)}%")
```

Run the script with 'python3 matrix.py'

This could print 'Percentage white: 25%' or report an error because there is "something missing" like the numpy module.

Do not 'pip install numpy' to install numpy because...

The advise is to use a virtual python environment.

That is an impressive name for a dedicated directory in which you put all modules for your scripts.
(You get a little bit more but that is the idea)

When you want to clean up later you simply delete the directory and all installed modules are gone and you have your python just like before you created the virtual environment.

So do the next:

```
    Create your virtual environment:
        python3 -m venv my_venv

    after this you run pip from your virtual environment :

        source path-to-venv
        pip install numpy
        deactivate

    that's all, or in a one-liner:

        on Linux   : my_venv/bin/pip install numpy
        on Windows : my_venv\Scripts\pip.exe install numpy

    you can run the script the same way:

        on Linux   : my_venv/bin/python3 matrix.py
        on Windows : my_venv\Scripts\python.exe matrix.py
        
Note the difference between Linux and Windows.

In the future I will only use the 'one-liner form for Linux syntax' but remember to use the Windows syntax on Windows.

I can not believe that I wrote the previous sentence but once said, you can not unsay things can you?
```

By now you can tell everybody in the universe that you created a neural network using a virtual environment. :smiley:

---

### Reading from a real picture

To be able to read a real picture file we need to create one.

So what does the inside of a picture image contain? At least some pixels with a color.

An RGB picture has 3 bytes per pixel (R, G, B an each with a value from 0 to 255).

255 means maximum intensity for that color of the pixel and 0 means no intensity at all for that color.

The values for white are 255, 255, 255 and for black 0, 0, 0

For the next step you need the PIL (Python Image Library) module which you can install with 'my_venv/bin/pip install pillow'.

Why install pillow while we need PIL? PIL support was stopped in 2011 and some developers forked PIL and named their fork pillow.

Create a perfect picture with the next python script 'create_picture.py' :

```python
from PIL import Image

# Create 4x4 RGB-picture, standard black
img = Image.new('RGB', (4, 4), color='black')
pixels = img.load()
# pixels is a 4x4 matrix with values like (R,G,B)
# The top left has x and y coordinates 0, 0
# This is the way to change the center 2x2 pixels to white
for y in range(1, 3):                   # rows 1 up to 3 (excludes 3!)
    for x in range(1, 3):               # columns 1 up to 3 (excludes 3!)
        pixels[x, y] = (255, 255, 255)  # change color to white

# Save as jpg
img.save('test.jpg', 'JPEG')
```

Run the script with 'my_venv/bin/python3 create_picture.py' to create the picture file.

Next create a script named test_picture.py :

```python
import numpy as np
from PIL import Image

# ----- Get the input pixels in P

# img has values between 0 and 255
img = Image.open('test.jpg')  # our RGB test picture, 4×4 picture assumed

# the inputs accept values between 0 and 1 so we divide by 255
P = np.array(img) / 255.0     # Normalize to [0,1]

# ----- Create the neuron and let it work

# set up W with a structure to match P
# to match it with the pixels this is a 4x4 matrix with values like (R,G,B)

# Before we started with weights of 1 and later 0.25 so we put 0.25 for every R G and B in the weights

W = np.array([[ [0,0,0], [0,0,0],          [0,0,0],          [0,0,0]],
              [ [0,0,0], [0.25,0.25,0.25], [0.25,0.25,0.25], [0,0,0]],
              [ [0,0,0], [0.25,0.25,0.25], [0.25,0.25,0.25], [0,0,0]],
              [ [0,0,0], [0,0,0],          [0,0,0],          [0,0,0]], ])

# whoopsie, when we have white in the center 
# the result will not be 4 * 0.25 = 1 
#                    but 4 * 0.25 * 3 (times 3 due to R, R and B )
# we correct this by dividing the weights by 3
# (remember this as 'another part of the training process' which we do later)

W = W / 3

# and the way to calculate the result with numpy is

R = np.sum(P * W)

# ----- Print the result

print(f"Percentage white: {R * 100.0:.2f}%")
```

Before you run the script read it to verify that it should output 'Percentage white: 100.00%'

After that run the script with 'my_venv/bin/python3 test_picture.py' to see 'Percentage white: 98.92%'

This difference is caused by the fact that computers do not calculate with an infinite accuracy.

A real neural network will output the 100% because it will adjust the weights during training to a slightly higher value.

Cool right?

Pictures in the real world are not perfect black and white so after the line:

P = np.array(img) / 255.0     # Normalize to [0,1]

we could use the next to correct almost black and almost white colors to perfect black and perfect white.

# here we correct almost black to perfect black

P[P < 0.42] = 0 # Threshold at 42% of 255

# the other part we correct to perfect white

P[P >= 0.42] = 1

---

## More complexity

We're not going to create a script with more complexity here, but let's think about larger neural networks and how they work.

We'll start with a network that still has the 4x4 picture as input, but now has an output for each possible position of our 2x2 white dot matrix.

We have 9 possible positions, so we want 9 outputs that tell us how white they are, just like the output we already have.

The 9 neuron neural network looks like this:

4 x 4 x 3 RGB inputs -- 48 connections --> to every one of the 9 outputs = 48 x 9 = 432 connections = 432 weights

....and we need to input these weights in our script. :cold_sweat:

When we mange to do that each output tells us how much of it is white and when you add the outputs together, the total comes to 1 (or should be 1 :smiley:).

That is possible, but the idea is that it becomes even more complicated when you want to analyze larger pictures.

Thhink of a real 16 megapixel picture.

Input = 16.777.216 pixels x 3 for RGB values = 50.331.648 values


To "divide and conquer" it is very common to use a multi-layered network, where the output of one layer of neurons is the input to the next layer.

Many inputs ----> layer 1 ----> layer 2 ----> last layer

The layer 1 can 'scan' the picture with small 3x3 windows to look for small details like little lines or colors and produce values indicating what it found in the windows. (note that I write windows and not window because it uses more windows in parallel to scan the picture)

It can create a new 'picture matrix' with lower dimensions than the original one and put its results in the new picture matrix.
These results have a meaning of little lines in certain directions or the major colors detected.

The layer 2 can 'scan' the 'new picture' with small 3x3 to look for small details like little figures or colors and produce values indicating what it found in the windows. ( note that this 3x3 window kind a checks a 9x9 window from the original input )

It can create a new 'picture matrix' with lower dimensions than its input and put its results in the new picture matrix.
These results have a meaning of little figures or the major colors detected.

And so on for additional layers.

The last layer receives the output of the previous layer and tries to recognize shapes, for example whether it is a picture of a cat or a dog.

This requires two outputs. One with a value between 0 and 1, representing the probability that it is a cat, and one with the probability that it is a dog.

This becomes very complicated to program with only numpy and pillow so we need some help here.

---

## Put together a real neural network

Before we put together a real network we need to know what we want with it and how to proceed.

### What do we want the network to do

We can program this example without a neural network, but we need something simple to get things explained.

Imagine a liquid level with values ​​between 0 and 100%, which we monitor with a camera that takes a picture every now and then.

I want to know two things about these pictures: the height of the level and the color of the liquid.

The neural network accepts pictures that are 100 pixels high and 25 pixels wide. (Pictures have different sizes, so we need to adjust the inputs.)

A picture looks like two rectangles of different colors on top of each other.

The top part is very light like white or almost white. The bottom part is perfect red, green or blue.

One thing I want is the height of the bottom part, and the other thing I want is the color of that bottom part.

When the valid colors are red, green, and blue, the network gets 3 outputs for that.

One output for the probability that the color is red, one for the probability that it is green, and one for the probability that it is blue.

The valid values ​​for percentage are between 1% and 100%. I'm skipping 0% because then we have no color, right?

This gives us another 100 results. One for each percentage. A total of 103 outputs.

What about the number of inputs by the way? 100 x 25 RGB cause 100 x 25 x 3 = 7500 

The output is 103 values so we end up with 7500 x 103 = 772.500 weights...

### What do we need to do

While numpy and pillow are already doing a great job, we need some extra help and that’s where the python module Keras flies in.

With Keras, we can create a neural network model and train it to process our input and give us the answers we need.

This Keras is complex to use and that’s where TensorFlow comes to the rescue.

We use TensorFlow to create the Keras model.

This model doesn’t have weights in its neurons, so we need to get them in somehow.

That process is called training the model and for that we need a lot of different examples that we will create.

The training is also done by TensorFlow.

To teach it to recognize the pictures we want, we need a lot of example pictures.

For each example picture we give, we also need to indicate to the training what the color is and what the percentage is.

we let the training use about 80% of the example pictures to generate the weights and put them into the model.

The other 20% of the sample pictures is used to validate the training result.

The training consists of several steps, which are called epochs.

Each epoch processes all of the examples and does this in batches of a size defined by us.

After each training batch a validation batch with seperarate examples is used to test the model.

Validation is just monitoring the progress and is not used to influence the training process.

After training, the definition of the model and the weights are saved in a file.

We can use this model file in a test script to analyse pictures and report the results.

Using the model is complicated so we call on TensorFlow for help again.

### Install some more python modules

To run the scripts we discuss below you need to install some python modules.

I install tensorflow-cpu because I do not have a GPU. :cry: When you have a GPU you can install tensorflow.
The module tensorflow-cpu does the same as tensorflow but uses the microprocessor of your computer and not a GPU.

'my_venv/bin/pip install tqdm tensorflow-cpu scikit-learn ai_edge_litert'

### Generate training data

The first step we take is to create training data. For this we **use the script script_generate_pictures.py from the code directory**.

This creates a large number of folders with names that contain the description of the content.

Example: **red_53** is a folder with pictures that are **red** for the bottom **53%**.

Each file in that folder has these properties and the file name is just a random number.

You may want to modify one line in the script to set the data_root directory to save the files to.

Then run the script with 'my_venv/bin/python3 script_generate_pictures.py'.

### Create and train the model

We use the script script_train_model.py  from the code directory to create and train the model.

You may need to modify one line in the script to set the data_root directory and maybe more when you made other changes in script_generate_pictures.py.

In short it reads the data, prepares data for training, creates an empty model, trains the model and saves 2 files.

It saves the model but also saves labels so the test script we use later can translate the output values, which are between 0 and 1, to text.

Before you dive into the code, which I am sure of you did not do yet...

```
You will find comments explaining a bit about the statements and more.

Like 2 active intermediate layers and a 3rd which you can activate by removing two # signs.

And also something that I have not been completely honest about....yet

Computers do not calculate with infinite accuracy. That I have been honest about already.

Remember math class at school where we say "y = ax + b is the equation of a straight line."
where "b" moves the line a little bit up or down?
Due to the not so infinite accuracy each neuron also uses a "b" to get better output results.
For the end result of every neuron there is a "b" which is the bias.

This makes the calculation for a neuron :
 - for each input : input result = input x weight
 - and end result : total result = sum of all input results + one neuron bias

In a matrix calculation this is:
 - a multiplication of the inputs and weights matrices + bias matrix
 - the result of multiplication of the inputs and weights matrices is a 1 x 1 matrix
 - which is a single number and just like that is the bias matrix just 1 single number
 - in our python scripts this could be : R = np.add(np.dot(P, W), B)
 - in the ideal world B would be [ 0 ]
```

Now you can I dive into the script or leave it for what it is and run it.

I use the time command to run the script to get a time report at the end:

(make sure to use a wide terminal window or resize it while the script runs to avoid a lot of lines)

```

time my_venv/bin/python3 script_train_model.py
:
a lot of output with long lines
:
real    1m52,437s
user    11m51,018s
sys     0m15,023s

```

After some 'sit back and relax' you have the trained model in the file model.keras and the encoded labels in label_encoders.pkl.

### Use the model

We use the script script_test_picture.py from the code directory to test with a picture we created before like:

'my_venv/bin/python3 script_test_picture.py <data_root>/red_78/638081.png'

Next create a 200 x 10 pixel white jpg file with an drawing program and make the bottom part red, green or blue and test.

Note that the input size is wrong. You can also use other colors in your jpg even when they are not in the model.

Test that and see that it will only report red, green or blue because it does not know the others.

<!--
This is the point where you may delete the complete dataset folder and put more colors in the script to generate pictures.

Just uncomment 1 line in script_generate_pictures.py.

Run the script to generate the pictures, run the script to create a new model and test your new colors.
-->

### Increase performance

The keras model model.keras can be converted to another model, which works 4 to 5 times faster.

Use the script script_convert_model_to_lite.py from the code directory for this.

"Be prepared" to see a lot of informational messages. "Don't panic".

Run 'my_venv/bin/python3 script_convert_model_to_lite.py' and you have a model.tflite.

### Use the faster model

We use the script script_test_picture_lite.py to test with a picture like:

'my_venv/bin/python3 script_test_picture_lite.py <data_root>/red_78/638081.png'

When that runs compare the time it takes :

'time my_venv/bin/python3 script_test_picture.py <data_root>/red_78/638081.png'

'time my_venv/bin/python3 script_test_picture_lite.py <data_root>/red_78/638081.png'

---

## Web application

You can use a web application to upload a picture and get the model outputs on the web page.

When you use it from a mobile you make a picture and upload that or upload a picture you made before.

I use a Raspberry Pi 3B and added apache and php after installing the bookworm OS.

The processing is slooooow but it works :smiley:

### Install the apache webserver

When you finished your basic install of your Raspberry Pi install the web server apache2.

Start a terminal with 'ssh pi@ip_address_of_your_Raspberry_Pi' and 'sudo apt install apache2 -y'

After this you should be able to use a web browser and browse to ```http://ip_address_of_your_Raspberry_Pi/``` and see a web page.

### Install PHP ( version 8.3 )

Installing php is a little more work.

(I followed <a href="https://php.watch/articles/php-8.3-install-upgrade-on-debian-ubuntu">https://php.watch/articles/php-8.3-install-upgrade-on-debian-ubuntu)</a>

Start a terminal with 'ssh pi@ip_address_of_your_Raspberry_Pi' and:
```
 - sudo apt install apt-transport-https
 - sudo curl -sSLo /usr/share/keyrings/deb.sury.org-php.gpg https://packages.sury.org/php/apt.gpg
 - sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/deb.sury.org-php.gpg] https://packages.sury.org/php/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/php.list'
 - sudo apt update
 - sudo apt install php8.3 php8.3-cli php8.3-{bz2,curl,mbstring,intl}
 - sudo apt install libapache2-mod-php8.3

Create a test file:
  - sudo -i
  - echo echo '<?php phpinfo(); ?>' > /var/www/html/info.php

```
Now you should be able to browse to  ```http://ip_address_of_your_Raspberry_Pi/info.php``` and see a page with info.

Note the filename in the 7th row **'Loaded Configuration File '**

You may increase some values in that file to allow bigger pictures to be uploaded like:

upload_max_filesize = 16M

post_max_size = 20M

activate the change :

 - when you run php8.3-fpm you first: sudo systemctl restart php8.3-fpm
 - sudo systemctl restart apache2.service

### Install the web application

We need to install some python modules in a virtual environment and after that the application.

Start a terminal with 'ssh pi@ip_address_of_your_Raspberry_Pi' and...

 - sudo -i
 - mkdir /var/www/html/level_detection
 - chown www-data: /var/www/html/level_detection
 - cd /var/www/html/level_detection

The next command will take 'some' time on a Raspberry Pi 3B.....

 - python3 -m venv my_venv
 - source my_venv/bin/activate
 - pip install numpy pillow ai_edge_litert scikit-learn
    - on the Raspberry Pi we use ai_edge_litert, not tensorflow
    - we need scikit-learn to read the pkl file, not to learn something
 - deactivate

Copy the next files to /var/www/html/level_detection

 - index.php
 - script_test_picture_lite.py
 - model.tflite
 - label_encoders.pkl
    - you may use my model.tflite and label_encoders,pkl or your own.

Now you should be able to browse to  ```http://ip_address_of_your_Raspberry_Pi/level_detection``` and use the application to analyse pictures.

---

## So what's next

Next could be:

 - dive into the **code_improved directory**
 - create, train and enjoy your own neural networks
 - maybe increase the number of colors in what you have now
 - look for answers to questions you have on the internet, in books, at friends
 - send me any comments you have on the above
 - relax and read a good guide 
 - and to find the the value of the answer to the question, count the letters of the answer:
    - it's the answer to life, the universe and everything

"So long, and thanks for all the fish."

Jack
