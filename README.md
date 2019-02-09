This repo detecated to the analysis at the Pima-Indians diabetes which 768 row and 9 columns at the columns are :

0- preg = Number of times pregnant

1- plas = Plasma glucose concentration in an oral glucose tolerance test

2- pres = Diastolic blood pressure (mm Hg)

3- skin = Triceps skin fold thickness (mm)

4- test = 2-Hour serum insulin (mu U/ml)

5- mass = Body mass index (weight in kg/ (height in m) ^2)

6- pedi = Diabetes pedigree function

7- age = Age (years)

8- class = Class variable (1: tested positive for diabetes, 0: tested negative for diabetes)


The implementation done by using Keras and the accuracy and loss results are: 
![accuracy](https://user-images.githubusercontent.com/23243761/52522051-a2c89080-2c80-11e9-8d0b-ce96458d1fb5.png)

![loss](https://user-images.githubusercontent.com/23243761/52522082-02bf3700-2c81-11e9-9f39-2e25b22ba034.png)

As we could see from this model(the photo of the model is attached bellow) we got 80.95% accuracy which is higher than any results which exists over the Internet:

![model](https://user-images.githubusercontent.com/23243761/52522133-7103f980-2c81-11e9-8802-5725e03df720.png)


After the definition of the inital model I retrain the last model and I got the improvment at the accuracy which is 81.39% and the loss accuracy is also improved compare to the previous model:

![8139_accuracy](https://user-images.githubusercontent.com/23243761/52522167-e1ab1600-2c81-11e9-9592-0f611bbe45c5.png)
![8139_loss](https://user-images.githubusercontent.com/23243761/52522169-e1ab1600-2c81-11e9-82df-edcfc0b91da1.png)
![final_model](https://user-images.githubusercontent.com/23243761/52522170-e1ab1600-2c81-11e9-9050-514ce8897989.png)



