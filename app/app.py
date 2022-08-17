import streamlit as st
import sklearn
import joblib,os
import numpy as np


def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    st.title("Cloud Machine Learning Project Demo Application")

    html_templ = """
	<div style="background-color:blue;padding:10px;">
	<h3 style="color:cyan">Simply select the values and use the models to make predictions</h3>
	</div>
	"""
    
    st.markdown(html_templ,unsafe_allow_html=True)

    activity = ["Number of wins determination", "Predict Score Per Minute", "About"]
    choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Number of wins determination":
        st.subheader("Number of wins determination")
        st.markdown("""
        This model shows where your number of wins should be, based on your level.
        If your actual number of wins is below the prediction, try harder. If your real wins are higher: Congrats! You are better than average, based on our dataset.
        """)
        level = st.number_input("What is your experience level in Call of Duty?",1,500)

        if st.button("Determination"):
            regressor = load_prediction_model("../models/win_level_regression.pkl")
            level_reshaped = np.array(level).reshape(-1,1)

			#st.write(type(experience_reshaped))
			#st.write(experience_reshaped.shape)

            predicted_wins = regressor.predict(level_reshaped)

            st.info("At level {}, your number of wins should be: {}".format(level,int((predicted_wins[0][0].round(0)))))
    elif choice == "Predict Score Per Minute":
        st.subheader("Predict Score Per Minute")
        st.markdown("""
        This model shows where your score per minute should be. If you are looking to farm score efficiently, this stat might be important for you.
        Simply enter your kdRatio and xp to find out your estimated score per minute.
        """)
        kdRatio = st.number_input(label="What is your kill/death ratio in Call of Duty?",step=1.,format="%.2f")
        xp = st.number_input("What is your overall experience points in Call of Duty?",1,5000000)

        if st.button("Predict Score Per Minute"):
            regressor = load_prediction_model("../models/predict_score_perminute.pkl")
            input_reshaped = [[kdRatio], [xp]]

            #st.write(type(experience_reshaped))
            #st.write(experience_reshaped.shape)

            predicted_score = regressor.predict(input_reshaped)

            st.info("At level {}, your score per minute should be: {}".format(kdRatio,int((predicted_score[0][0].round(0)))))        
    else:
        st.subheader("About")
        st.markdown("""
			## Cloud Machine Learning Project Demo Application
			
			##### By
			+ **[Matyas Greff - x21129878](https://github.com/MatyasGreff)**
			+ **[Jeisse Rocha - x21129924](https://github.com/Jeisse)**
			""")
        st.markdown("""
			## Repo
			
			+ **[cml-project-sem3](https://github.com/MatyasGreff/cml-project-sem3)**
			""")
if __name__ == "__main__":
    main()