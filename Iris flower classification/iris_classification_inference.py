import pickle
import os
import pandas as pd 
import numpy as np

class Inference:
        def __init__(self, model_path):
            self.model_path = model_path
            

            if os.path.exists(self.model_path):
                self.model = pickle.load(open(self.model_path, "rb"))
                
            else:
                print("Model Path is no correct!!")

        def users_input(self):
            print("Enter the correct information to predict the correct classification of Iris flower")
            
            while True:
                try:
                    sepal_l = float(input("Enter the Sepal Length : "))
                    sepal_w = float(input("Enter the Sepal Width : "))
                    petal_l = float(input("Enter the Petal Length : "))
                    petal_w = float(input("Enter the Petal width : "))
                    
                    # If all inputs are successfully converted to float, break the loop
                    break
                except ValueError:
                    print("Invalid input. Please enter numeric values.")

            u_input_list = [sepal_l, sepal_w, petal_l, petal_w]
            feature_name = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            df_u_input = pd.DataFrame([u_input_list], columns=feature_name)

            return df_u_input
        
        def prediction(self):
             df = self.users_input()
             prediction = self.model.predict(df)
             return prediction
        


if __name__ == "__main__":

    ml_model_path = r"D:\Data Science\ML Project\Iris flower classification\k_neighborsclassifier.pkl"
    
    inference = Inference(ml_model_path)

    pred = inference.prediction()
    predicted_value = round(pred.tolist()[0])

    if predicted_value ==0:
        print(f"Predicted classification of Iris flower according to your input on Sepal Length, sepal width, petal length and petal width is : Iris-setosa")

    elif predicted_value ==1:
        print(f"Predicted classification of Iris flower according to your input on Sepal Length, sepal width, petal length and petal width is : Iris-versicolor")
    elif predicted_value ==2:
        print(f"Predicted classification of Iris flower according to your input on Sepal Length, sepal width, petal length and petal width is : Iris-virginica")

    else:
        print("Your Input is not valid")

