import streamlit as st
import sklearn
import joblib,os
import numpy as np


def load_Prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    st.title("Cloud Machine Learning Project Demo Application")

    html_templ = """
	<div style="background-color:blue;padding:10px;">
	<h3 style="color:cyan">Simply select the values and use the models to make Predictions</h3>
	</div>
	"""
    
    st.markdown(html_templ,unsafe_allow_html=True)

    activity = ["Number of wins Prediction", "Predict Score Per Minute", "From kill ratio, predict Win Ratio", "Predict Duo Win Ratio", "About"]
    choice = st.sidebar.selectbox("Menu", activity)

    if choice == "Number of wins Prediction":
        st.subheader("Number of wins Prediction")
        st.markdown("""
        This model shows where your number of wins should be, based on your level.
        If your actual number of wins is below the Prediction, try harder. If your real wins are higher: Congrats! You are better than average, based on our dataset.
        """)
        level = st.slider("What is your experience level in Call of Duty?",1,500, value=100)

        if st.button("Prediction"):
            regressor = load_Prediction_model("models/win_level_regression.pkl")
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
        kdRatio = st.slider(label="What is your kill/death ratio in Call of Duty?",step=float(0.01), min_value=float(0.0), max_value=float(10.0), value=1.0)
        xp = st.number_input("What is your overall experience points in Call of Duty?",1,5000000)

        if st.button("Predict Score Per Minute"):
            regressor = load_Prediction_model("models/predict_score_perminute.pkl")
            input_reshaped = [[kdRatio, xp]]

            #st.write(type(experience_reshaped))
            #st.write(experience_reshaped.shape)

            predicted_score = regressor.predict(input_reshaped)

            st.info("With a {} kill/death ratio, and {} total xp, your score per minute is: {}".format(kdRatio, xp, (predicted_score[0][0].round(2))))
    elif choice == "From kill ratio, predict Win Ratio":
        st.subheader("From kill ratio, predict Win Ratio")
        st.markdown("""
        This model shows what your win ratio should be, based on your kill/death ratio.
        If your actual number of wins is below the Prediction, try harder. If your real wins are higher: Congrats! You are better than average, based on our dataset.
        """)
        kdRatio = st.slider(label="What is your kill/death ratio in Fortnite?",step=float(0.01), min_value=float(0.0), max_value=float(10.0), value=1.0)

        if st.button("Prediction"):
            regressor = load_Prediction_model("models/givenKillRatioPredictWinRatio.pkl")
            kd_reshaped = np.array(kdRatio).reshape(-1,1)

            #st.write(type(experience_reshaped))
            #st.write(experience_reshaped.shape)

            predicted_wins = regressor.predict(kd_reshaped)

            st.info("With a {} kill/death ratio, your win ratio should be: {}".format(kdRatio,(predicted_wins[0].round(2))))
    elif choice == "Predict Duo Win Ratio":
        st.subheader("Predict Duo Win Ratio")
        st.markdown("""
        This model shows what your duo win ratio should be, based on your solo win ratio.
        If your actual number of wins is below the Prediction, try harder. If your real wins are higher: Congrats! You are better than average, based on our dataset.
        """)
        solowin = st.slider(label="What is your win ratio when playing on duos in Fortnite?",step=float(0.01), min_value=float(0.0), max_value=float(10.0), value=1.0)

        if st.button("Prediction"):
            regressor = load_Prediction_model("models/givenSoloWinRatioPredictDuoWinRatio.pkl")
            solowin_reshaped = np.array(solowin).reshape(-1,1)

            #st.write(type(experience_reshaped))
            #st.write(experience_reshaped.shape)

            predicted_wins = regressor.predict(solowin_reshaped)

            st.info("With a {} solo win ratio, your duo win ratio should be: {}".format(solowin,(predicted_wins[0].round(2))))         
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
