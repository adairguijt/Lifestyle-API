#create basic streamlit structure 

#import packages
import plotly.graph_objects as go 

import streamlit as st
import pandas as pd
import joblib
import openai

with open("openai.txt", "r") as file: 
    key = file.readline().strip()

client = openai.OpenAI(api_key= key)

GPTmodel = "gpt-4.1-mini-2025-04-14" 


def generate_ai_lifestyle_advice(score, recommendations, user_answers):

    prompt = f"""

    ###CONTEXT###
    The user's lifestyle score is {score}.

    Score meaning:
    - Below 500 = unhealthy
    - 500-650 = average
    - 650-750 = healthy
    - Above 750 = exceptional

    Recommendations:
    {recommendations}

    User answers:
    {user_answers}

    ###TASK###

    Write a short personalized lifestyle suggestion using: 
    -the information from the lifestyle score and recommendations 
    -knowledge found on the internet about life satisfaction and wellbeing
     
    
    Suggest a website that the user can use to integrate the lifestyle suggestion in their life. 
    
    ###TARGETS###
    The lifestyle advice should help the user to live a more healthy and happy life.
    The text should encourage the user to make use of the website to improve their lifestyle.

    ###CONSTRAINTS###
    The information from the lifestyle score and recommendations should not be copied in the answer.
    It should have a max of 200 words. 
    """



    response = client.chat.completions.create(
        model= GPTmodel,
        messages=[
            {"role": "system", "content": "You are a helpful lifestyle coach."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

model = joblib.load("lifestyle_model.pkl")





#copy actionable features into the streamlit apy
actionable_features = {
    "SUPPORTING_OTHERS": {"direction": "increase", "step": 1, "min" : 0, "max": 10},
    "TODO_COMPLETED": {"direction": "increase", "step": 1, "min" : 0, "max": 10},
    "TIME_FOR_PASSION": {"direction": "increase", "step": 1, "min" : 0, "max": 10},
    "SUFFICIENT_INCOME": {"direction": "target", "step": 1, "target": 2, "min" : 1, "max" : 2},
    "CORE_CIRCLE": {"direction": "increase", "step": 1, "min" : 0, "max": 10},
    "WEEKLY_MEDITATION": {"direction": "increase", "step": 1, "min" : 0, "max": 10},
    "DAILY_STEPS": {"direction": "increase", "step": 1, "min" : 1, "max": 10},
    "LIFE_VISION": {"direction": "increase", "step": 1, "min" : 0, "max": 10},
    "DONATION": {"direction": "increase", "step": 1, "min" : 0, "max": 5},
    "BMI_RANGE": {"direction": "target", "step": 1, "target": 1, "min" : 1, "max" : 2},
    "FRUITS_VEGGIES": {"direction": "increase", "step": 1, "min" : 0, "max": 5},
    "DAILY_STRESS": {"direction": "decrease", "step": 1, "max" : 5, "min": 0},
    "DAILY_SHOUTING": {"direction": "decrease", "step": 1, "max" : 10, "min": 0}
}

#generateAIfunction 


#copy function definition of recommendation engine
def generate_optimized_advice(user_answers, model, actionable_features, top_n=3):
  #predict current score
  current_input = pd.DataFrame([user_answers])
  current_score = model.predict(current_input)[0]
  #empty recommendation list
  recommendations = []
  #loop through all top actionable habits
  for feature, rule in actionable_features.items():
    #skip missing features
    if feature not in user_answers:
      continue
    #simulation copy for experimental changes
    new_user = user_answers.copy()
    #get current value
    current_value = user_answers[feature]
    #modify the habit
    if rule["direction"] == "increase":
      new_value = min(current_value + rule["step"], rule["max"])
    elif rule["direction"] == "decrease":
      new_value = max(current_value - rule["step"], rule["min"])
    elif rule["direction"] == "target":
      target = rule["target"]

      if current_value < target:
        new_value = current_value + rule["step"]
      elif current_value > target:
        new_value = current_value - rule["step"]
      else:
        continue
    #skip unchanged values
    if new_value == current_value:
            continue
    #apply temporary change
    new_user[feature] = new_value
    #predict new score
    new_input = pd.DataFrame([new_user])
    new_score = model.predict(new_input)[0]
    #calculate improvement
    improvement = new_score - current_score
    #feature range
    feature_range =rule["max"] - rule ["min"]
    #calculate intervention size 
    change_size = abs(new_value - current_value)
    #convert to fraction of total size 
    range_fraction = change_size / feature_range 
    #normalize improvement
    normalized_improvement = improvement / range_fraction
    #store recommendation
    if improvement > 0:
      recommendations.append({
          "feature": feature,
          "current_value": current_value,
          "suggested_value": new_value,
          "estimated_score_gain": round(float(improvement), 2),
          "normalized_gain": round(float(normalized_improvement), 2),
          "new_estimated_score": round(float(new_score), 2)
            })
    #sort recommendations
  recommendations = sorted(
      recommendations,
      key=lambda x: x["normalized_gain"],
      reverse=True
  )
  #return results
  return {
        "current_lifestyle_score": round(float(current_score), 2),
        "best_recommendations": recommendations[:top_n]
  }





##Create layout of different pages using streamlit
import streamlit as st

# initialize page system
if "page" not in st.session_state:
    st.session_state.page = 1

# navigation functions
def next_page():
    st.session_state.page += 1

def previous_page():
    st.session_state.page -= 1


# ---------------- PAGE 1 ----------------

if st.session_state.page == 1:

    st.title("Lifestyle Score Predictor")

    st.write("""
    Welcome to the Lifestyle Score Predictor.

    This application uses machine learning to estimate your predicted lifestyle score
    based on your daily habits, mental wellbeing, health, social relationships,
    and productivity patterns.

    At the end, you will receive:
    - Your predicted lifestyle score
    - 3 personalized recommendations
    - Estimated lifestyle improvements based on your habits

    The questionnaire takes approximately 2–3 minutes.
    """)

    st.button("Start", on_click=next_page)

    # ---------------- PAGE 2 ----------------

elif st.session_state.page == 2:

    st.title("Personal Information")

    st.write("""
    First, we will collect some basic personal information.
    This helps the model better estimate your lifestyle score.
    """)

    # Gender
    st.session_state["GENDER"] = st.radio(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Male" if x == 0 else "Female",
        index=st.session_state.get("GENDER", 0)
    )

      # Age Group options
    age_options = ["<20", "21-35", "36-50", "51+"]

    # get previously selected age group
    current_age = st.session_state.get("AGE_GROUP", "21-35")

    # create selectbox and store answer
    st.session_state["AGE_GROUP"] = st.selectbox(
    "Age Group",
    age_options,
    index=age_options.index(current_age)
  )

  # convert selected age group into one-hot encoding
    AGE_36_TO_50 = 1 if st.session_state["AGE_GROUP"] == "36-50" else 0
    AGE_51_OR_MORE = 1 if st.session_state["AGE_GROUP"] == "51+" else 0
    AGE_LESS_THAN_20 = 1 if st.session_state["AGE_GROUP"] == "<20" else 0

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back", on_click=previous_page)

    with col2:
        st.button("Next", on_click=next_page)


# ---------------- PAGE 3 ----------------

elif st.session_state.page == 3:

    st.title("Health")

    st.write("""
    These questions focus on your physical lifestyle habits, such as nutrition,
    sleep, body health, and daily movement.
    """)

    st.session_state["FRUITS_VEGGIES"] = st.slider(
        "How many fruits or vegetables do you eat every day?",
        0, 5,
        st.session_state.get("FRUITS_VEGGIES", 2)
    )

    st.session_state["SLEEP_HOURS"] = st.slider(
        "How many hours do you typically sleep per night?",
        1, 10,
        st.session_state.get("SLEEP_HOURS", 5)
    )

    st.session_state["BMI_RANGE"] = st.radio(
        "What is your BMI range?",
        options=[1, 2],
        format_func=lambda x: "Healthy / Normal" if x == 1 else "Unhealthy / Abnormal",
        index=0 if st.session_state.get("BMI_RANGE", 1) == 1 else 1
    )

    st.session_state["DAILY_STEPS"] = st.slider(
        "How many steps, in thousands, do you typically walk every day?",
        1, 10,
        st.session_state.get("DAILY_STEPS", 5)
    )

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back", on_click=previous_page)

    with col2:
        st.button("Next", on_click=next_page)



# ---------------- PAGE 4 ----------------

elif st.session_state.page == 4:
    st.title("Mental Wellbeing")

    st.write("""
    These questions focus on stress, flow, meditation, and time spent on meaningful activities.
    """)

    st.session_state["DAILY_STRESS"] = st.slider(
       "How much stress do you typically experience in a day?",
        0, 5,
        st.session_state.get("DAILY_STRESS", 2)
    )

    st.session_state["FLOW"] = st.slider(
        "In a typical day, how many hours do you experience flow?",
        0, 10,
        st.session_state.get("FLOW", 5)
    )

    st.session_state["TIME_FOR_PASSION"] = st.slider(
        "How many hours do you spend per day doing what you are passionate about?",
        0, 10,
        st.session_state.get("TIME_FOR_PASSION", 5)
    )

    st.session_state["WEEKLY_MEDITATION"] = st.slider(
        "How often a week do you sit down to meditate?",
        0, 10,
        st.session_state.get("WEEKLY_MEDITATION", 5)
    )

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
      st.button("Back", on_click=previous_page)

    with col2:
      st.button("Next", on_click=next_page)

# ---------------- PAGE 5 ----------------

elif st.session_state.page == 5:

    st.title("Social Life")

    st.write("""
    These questions focus on your close relationships, daily interactions,
    support for others, and emotional communication.
    """)

    st.session_state["CORE_CIRCLE"] = st.slider(
        "How many people are you very close to?",
        0, 10,
        st.session_state.get("CORE_CIRCLE", 5)
    )

    st.session_state["SUPPORTING_OTHERS"] = st.slider(
        "How many people do you help live a better life?",
        0, 10,
        st.session_state.get("SUPPORTING_OTHERS", 5)
    )

    st.session_state["SOCIAL_NETWORK"] = st.slider(
        "How many people do you actively interact with during a day?",
        0, 10,
        st.session_state.get("SOCIAL_NETWORK", 5)
    )

    st.session_state["DAILY_SHOUTING"] = st.slider(
        "How often do you shout or sulk at someone on an average day?",
        0, 10,
        st.session_state.get("DAILY_SHOUTING", 5)
    )

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back", on_click=previous_page)

    with col2:
        st.button("Next", on_click=next_page)

# ---------------- PAGE 6 ----------------

elif st.session_state.page == 6:

    st.title("Productivity & Stability")

    st.write("""
    These questions focus on productivity, financial stability,
    long-term vision, and contribution to others.
    """)

    st.session_state["TODO_COMPLETED"] = st.slider(
        "How well do you complete your weekly To-Do list?",
        0, 10,
        st.session_state.get("TODO_COMPLETED", 5)
    )

    st.session_state["DONATION"] = st.slider(
        "How many times do you donate your time or money to good causes per month?",
        0, 5,
        st.session_state.get("DONATION", 2)
    )

    st.session_state["SUFFICIENT_INCOME"] = st.radio(
        "Is your income sufficient to cover basic life expenses?",
        options=[1, 2],
        format_func=lambda x: "No" if x == 1 else "Yes",
        index=0 if st.session_state.get("SUFFICIENT_INCOME", 1) == 1 else 1
    )

    st.session_state["LIFE_VISION"] = st.slider(
        "For how many years ahead do you have a clear vision of your life?",
        0, 10,
        st.session_state.get("LIFE_VISION", 5)
    )

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back", on_click=previous_page)

    with col2:
        st.button("See My Results", on_click=next_page)

# ---------------- PAGE 7 ----------------

elif st.session_state.page == 7:

    st.title("Your predicted lifestyle score")

    #collect all inputs into user answers 
    user_answers = {
        "FRUITS_VEGGIES": st.session_state["FRUITS_VEGGIES"],
        "DAILY_STRESS": st.session_state["DAILY_STRESS"],
        "CORE_CIRCLE": st.session_state["CORE_CIRCLE"],
        "SUPPORTING_OTHERS": st.session_state["SUPPORTING_OTHERS"],
        "SOCIAL_NETWORK": st.session_state["SOCIAL_NETWORK"],
        "DONATION": st.session_state["DONATION"],
        "BMI_RANGE": st.session_state["BMI_RANGE"],
        "TODO_COMPLETED": st.session_state["TODO_COMPLETED"],
        "FLOW": st.session_state["FLOW"],
        "DAILY_STEPS": st.session_state["DAILY_STEPS"],
        "LIFE_VISION": st.session_state["LIFE_VISION"],
        "SLEEP_HOURS": st.session_state["SLEEP_HOURS"],
        "DAILY_SHOUTING": st.session_state["DAILY_SHOUTING"],
        "SUFFICIENT_INCOME": st.session_state["SUFFICIENT_INCOME"],
        "TIME_FOR_PASSION": st.session_state["TIME_FOR_PASSION"],
        "WEEKLY_MEDITATION": st.session_state["WEEKLY_MEDITATION"],
        "GENDER": st.session_state["GENDER"],
        "AGE_36_TO_50": 1 if st.session_state["AGE_GROUP"] == "36-50" else 0,
        "AGE_51_OR_MORE": 1 if st.session_state["AGE_GROUP"] == "51+" else 0,
        "AGE_LESS_THAN_20": 1 if st.session_state["AGE_GROUP"] == "<20" else 0
    }

    st.session_state["user_answers"] = user_answers

    result = generate_optimized_advice(
        user_answers=user_answers,
        model=model,
        actionable_features=actionable_features,
        top_n=3
    )

    st.session_state["result"] = result

    prediction = st.session_state["result"]["current_lifestyle_score"]

    # determine color
    if prediction < 500:
        score_color = "red"

    elif prediction < 650:
        score_color = "orange"

    elif prediction < 750:
        score_color = "green"
    else:
        score_color = "darkgreen"

    # display title
    st.subheader("Your Predicted Lifestyle Score")

    # rainbow score for 750+
    
    st.markdown(
        f"""
        <h1 style='
            font-size:80px;
            text-align:center;
            color:{score_color};
            font-weight:bold;
        '>
            {prediction:.2f}
        </h1>
        """,
        unsafe_allow_html=True
    )

    score = st.session_state["result"]["current_lifestyle_score"]
   

    if score < 500:
       st.error("Your lifestyle score is currently quite low. There may be several areas in your daily habits that could strongly benefit from improvement.")

    elif score < 650:
       st.warning("Your lifestyle score is moderate. You already have some healthy habits, but there are still important opportunities for growth and balance.")

    elif score < 750:
       st.success("Your lifestyle score is good. You seem to maintain several healthy and supportive lifestyle habits.")

    else:
       st.success("Excellent lifestyle score. Your habits and routines appear to support a very healthy and fulfilling lifestyle.")
    
    
   
    def create_radar_chart(user_answers):
        health_score = (
            (user_answers["FRUITS_VEGGIES"] / 5) +
            (user_answers["SLEEP_HOURS"] / 10) +
            (user_answers["DAILY_STEPS"] / 10) +
            (1 if user_answers["BMI_RANGE"] == 1 else 0)
        ) / 4 * 100

        mental_score = (
            ((5 - user_answers["DAILY_STRESS"]) / 5) +
            (user_answers["FLOW"] / 10) +
            (user_answers["TIME_FOR_PASSION"] / 10) +
            (user_answers["WEEKLY_MEDITATION"] / 10)
        ) / 4 * 100

        social_score = (
            (user_answers["CORE_CIRCLE"] / 10) +
            (user_answers["SUPPORTING_OTHERS"] / 10) +
            (user_answers["SOCIAL_NETWORK"] / 10) +
            ((10 - user_answers["DAILY_SHOUTING"]) / 10)
        ) / 4 * 100

        productivity_score = (
            (user_answers["TODO_COMPLETED"] / 10) +
            (user_answers["DONATION"] / 5) +
            (1 if user_answers["SUFFICIENT_INCOME"] == 2 else 0) +
            (user_answers["LIFE_VISION"] / 10)
        ) / 4 * 100

        categories = [
            "Health",
            "Mental Wellbeing",
            "Social Life",
            "Productivity & Stability"
        ]

        values = [
            health_score,
            mental_score,
            social_score,
            productivity_score
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Lifestyle Profile"
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False
        )

        return fig
    
    st.subheader("Your Lifestyle Profile")

    radar_fig = create_radar_chart(st.session_state["user_answers"])
    st.plotly_chart(radar_fig, use_container_width= True)

    st.write("""
    For advice on improving your lifestyle habits and help create a more balanced, and more fulfilling life, click below:
    """)

    if st.button("Show My Lifestyle Recommendations"):
        st.session_state.page = 8
        st.rerun()


    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back", on_click=previous_page)

# ---------------- PAGE 8 ----------------

elif st.session_state.page == 8:

    st.title("Your Lifestyle Recommendations")

    if "result" not in st.session_state:
        st.warning("Please calculate your lifestyle score first.")
        st.button("Back", on_click=previous_page)

    else:
        result = st.session_state["result"]

        st.subheader("Top 3 Suggested Changes")


        for rec in result["best_recommendations"]:
            
            recommendation_text = {

                "FRUITS_VEGGIES": {
                    "title": "🥗 Improve your nutrition",
                    "description": "Try increasing your daily intake of fruits and vegetables from {current} to {suggested} portions per day. This may improve physical health and wellbeing",
                    "why": "A nutrient-rich diet supports physical health, energy levels, mood regulation, and long-term wellbeing."
                },

                "DAILY_STRESS": {
                    "title": "🧠 Reduce daily stress",
                    "description": "Reducing your daily stress level from {current} to {suggested} may significantly improve mental wellbeing and emotional balance.",
                    "why": "Chronic stress is strongly linked to reduced wellbeing, burnout, sleep problems, and emotional instability."
                },

                "CORE_CIRCLE": {
                    "title": "❤️ Strengthen close relationships",
                    "description": "Building deeper connections with a few more trusted people from {current} to {suggested} may improve emotional support and overall happiness.",
                    "why": "Strong close relationships are among the strongest predictors of long-term life satisfaction and wellbeing."
                },

                "SUPPORTING_OTHERS": {
                    "title": "🤝 Help others more often",
                    "description": "Increasing the number of people you positively support from {current} to {suggested} may improve purpose and fulfillment.",
                    "why": "Acts of kindness and compassion are strongly associated with meaning, happiness, and emotional wellbeing."
                },

                "SOCIAL_NETWORK": {
                    "title": "💬 Increase social interaction",
                    "description": "Try increasing your daily social interactions from {current} to {suggested}. This may improve emotional wellbeing",
                    "why": "Healthy social interaction can reduce loneliness and improve emotional resilience and wellbeing."
                },

                "DONATION": {
                    "title": "🌍 Contribute to meaningful causes",
                    "description": "Increasing how often you donate your time or resources from {current} to {suggested} times a month may improve feelings of purpose and connection.",
                    "why": "Generosity and contribution are linked to increased life meaning and emotional wellbeing."
                },

                "BMI_RANGE": {
                    "title": "⚕️ Improve physical health balance",
                    "description": "Moving toward a healthier BMI range may positively affect energy, health, and overall lifestyle balance.",
                    "why": "Maintaining a healthy body composition is associated with improved physical and mental health outcomes."
                },

                "TODO_COMPLETED": {
                    "title": "🚀 Improve productivity habits",
                    "description": "Improving your weekly task completion from {current} to {suggested} may create more structure and satisfaction.",
                    "why": "Consistent productivity and organization can reduce stress and improve self-confidence and stability."
                },

                "FLOW": {
                    "title": "🌊 Increase flow experiences",
                    "description": "Try increasing the amount of time you experience flow from {current} to {suggested} hours per day. This may create more fulfillment in life.",
                    "why": "Flow states are strongly associated with fulfillment, creativity, intrinsic motivation, and happiness."
                },

                "DAILY_STEPS": {
                    "title": "🚶 Increase daily movement",
                    "description": "Try increasing your walking from approximately {current}k to {suggested}k steps per day. This might improve physical health.",
                    "why": "Regular physical movement supports cardiovascular health, mood, energy, and long-term wellbeing."
                },

                "LIFE_VISION": {
                    "title": "🎯 Develop a clearer life vision",
                    "description": "Expanding your long-term vision and planning horizon from {current} to {suggested} years may improve direction and motivation.",
                    "why": "Having a meaningful future vision is strongly connected to purpose, motivation, and psychological wellbeing."
                },

                "SLEEP_HOURS": {
                    "title": "😴 Improve sleep habits",
                    "description": "Adjusting your sleep from {current} to {suggested} hours per night may improve recovery and mental clarity.",
                    "why": "Healthy sleep is one of the strongest predictors of emotional balance, cognitive performance, and health."
                },

                "DAILY_SHOUTING": {
                    "title": "🕊️ Improve emotional communication",
                    "description": "Reducing moments of shouting or emotional conflict from {current} to {suggested} may improve relationship quality and emotional balance.",
                    "why": "Frequent emotional conflict can negatively affect both relationships and long-term wellbeing."
                },

                "SUFFICIENT_INCOME": {
                    "title": "💰 Improve financial stability",
                    "description": "Improving financial security and meeting basic life expenses may reduce stress and improve stability.",
                    "why": "Financial stability is strongly associated with reduced stress and improved overall wellbeing."
                },

                "TIME_FOR_PASSION": {
                    "title": "🔥 Spend more time on meaningful activities",
                    "description": "Increasing time spent on activities you love from {current} to {suggested} hours per day may improve life satisfaction.",
                    "why": "Passion and meaningful engagement are closely linked to fulfillment, motivation, and happiness."
                },

                "WEEKLY_MEDITATION": {
                    "title": "🧘 Deepen your meditation practice",
                    "description": "Increasing your weekly meditation frequency from {current} to {suggested} sessions may improve calmness and mental clarity.",
                    "why": "Meditation is associated with reduced stress, emotional regulation, attention, and psychological wellbeing."
                }
            }

            

            feature_info = recommendation_text[rec["feature"]]

            gain = rec["estimated_score_gain"]

            if gain >= 7:
                impact = "Very High Impact"
            elif gain >= 4:
                impact = "High Impact"
            elif gain >= 2:
                impact = "Moderate Impact"
            else:
                impact = "Small Impact"

            

            with st.container(border=True):
                st.subheader(feature_info["title"])

                

                st.markdown(
                    f"""
                    <div style="
                        font-size:20px;
                        font-weight:500;
                        margin-bottom:20px;
                    ">
                        {feature_info["description"].format(
                            current=rec["current_value"],
                            suggested=rec["suggested_value"]
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.info(f"Impact Level: {impact}")

                st.success(
                    f"Estimated score gain: +{rec['estimated_score_gain']} points"
                )

                st.caption(feature_info["why"])
            
           
            
        

        st.write("---")

        col1, col2 = st.columns(2)

        with col1:
            st.button("Back", on_click=previous_page)

        with col2: 
            if st.button("Generate Personal AI Advice"):
                st.session_state.page = 9
                st.rerun()

        with col2:
            if st.button("Start Again"):
                st.session_state.clear()
                st.session_state.page = 1
                st.rerun()

# ---------------- PAGE 9 ----------------

elif st.session_state.page == 9:

    st.title("Your Personal AI Lifestyle Advice")

    # Generate advice once
    if "ai_advice" not in st.session_state:

        ai_advice = generate_ai_lifestyle_advice(
            score=st.session_state["result"]["current_lifestyle_score"],
            recommendations=st.session_state["result"]["best_recommendations"],
            user_answers=st.session_state["user_answers"]
        )

        st.session_state["ai_advice"] = ai_advice

    # Display advice
    st.write(st.session_state["ai_advice"])

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.button("Back", on_click=previous_page)

    with col2:
        if st.button("Start Again"):
            st.session_state.clear()
            st.session_state.page = 1
            st.rerun()


