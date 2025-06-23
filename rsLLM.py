#-----------------------------------------------------------------------------------
# Integration: Intelligent Agents + Recommendation System + LLM (gpt-4o-mini)
# Version 1.0
#-----------------------------------------------------------------------------------

import os
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
import streamlit as st  # type: ignore
from openai import ChatCompletion
import datetime
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from surprise import Dataset, Reader, KNNBasic #type: ignore
from surprise.model_selection import train_test_split #type: ignore
from dotenv import load_dotenv #type: ignore

# API Key loaded in .env file

load_dotenv()


# API Configurations
os.environ["OPENAI_API_KEY"] = "your OpenAI API Key"

# ----------------------------------------------------------------------------------------
# Tool 1: Generate Recommendations Using KNN
# ----------------------------------------------------------------------------------------

@tool
def get_knn_recommendations(csv_path: str, user_id: str, top_n: int = 5) -> dict:
    """
    Generates recommendations using KNN based on the CSV data.

    Args:
        csv_path (str): Path to the CSV file.
        user_id (int): User ID for recommendations.
        top_n (int): Number of recommendations to generate.

    Returns:
        dict: User data and recommendations.
    """
    # Load the dataset
    dataset = pd.read_csv(csv_path)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(dataset[['UserID', 'ProductID', 'Rating']], reader)
    
    # Split the data into train and test sets
    trainset, _ = train_test_split(data, test_size=0.2)
    
    # Train the KNN model
    sim_options = {'name': 'pearson', 'user_based': True}
    algo = KNNBasic(k=10, sim_options=sim_options)
    algo.fit(trainset)
    
    # Get products not rated by the user
    user_data = dataset[dataset['UserID'] == user_id]
    if user_data.empty:
        return {"error": f"User {user_id} not found in the dataset."}
    
    all_products = dataset['ProductID'].unique()
    rated_products = user_data['ProductID'].unique()
    unrated_products = [item for item in all_products if item not in rated_products]
    
    # Predict ratings for unrated products
    predictions = [algo.predict(user_id, item) for item in unrated_products]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
    
    recommendations = []
    for pred in top_predictions:
        # Filter product information
        product_data = dataset[dataset['ProductID'] == pred.iid][['ProductID', 'Category', 'FabricType', 'Style', 'ColorPalette', 'Season']].drop_duplicates()
        if not product_data.empty:
            recommendations.append(product_data.iloc[0].to_dict())
    
    user_info = {
        "UserID": user_id,
        "Age": user_data['Age'].iloc[0],
        "Gender": user_data['Gender'].iloc[0],
        "BodyType": user_data['BodyType'].iloc[0],
    }
    
    return {
        "user_data": user_info,
        "recommendations": recommendations
    }


# ----------------------------------------------------------------------------------------
# Tool 2: Configure Tool for Agents to Interact with LLM via LangChain
# ----------------------------------------------------------------------------------------

@tool
def query_llm(user_data: dict, recommendations: list, current_season: str) -> str:
    """
    Queries an LLM to enrich recommendations with current trends and new styles.

    Args:
        user_data (dict): User data.
        recommendations (list): Initial recommendations list.
        current_season (str): Current season.

    Returns:
        str: Recommendations enriched by the LLM.
    """
    # Initialize the OpenAI model via LangChain
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Model used
        #model="o1",
        temperature=0.7,  # Controls creativity
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create the input prompt
    prompt = (
        f"You are a fashion expert. Based on the customer data:\n"
        f"- Age: {user_data['Age']}\n"
        f"- Gender: {user_data['Gender']}\n"
        f"- Body Type: {user_data['BodyType']}\n"
        f"- Current Season: {current_season}\n\n"
        f"And the initial recommendations:\n"
    )
    for rec in recommendations:
        prompt += (
            f"- Product {rec['ProductID']}: Category {rec['Category']}, Style {rec['Style']}, "
            f"Fabric {rec['FabricType']}, Color {rec['ColorPalette']}, Season {rec['Season']}\n"
        )
    prompt += (
        "\nPlease enrich these recommendations considering current trends, "
        "new styles, and innovative products that match the customer profile. "
        "Add details about why each item is suitable."
    )

    # Send the prompt to the model and get the response
    response = llm([HumanMessage(content=prompt)])
    
    # Return the response generated by the LLM
    return response.content


# -------------------------------------------------
# Agent Configuration
# -------------------------------------------------

# Data Analyst Agent
data_analyst = Agent(
    role="Data Analyst",
    goal="Generate initial recommendations for user {user_id} using the KNN model.",
    backstory="You are a data analysis and recommendation systems expert.",
    tools=[get_knn_recommendations],
    memory=False,
    verbose=True
)

# Advanced Recommendation Generator Agent
recommendation_generator_llm = Agent(
    role="Advanced Recommendation Generator",
    goal="Enrich recommendations for user {user_id} using current fashion trends and additional information.",
    backstory="You are a fashion expert using an LLM to improve recommendations.",
    tools=[query_llm],
    memory=True,
    verbose=True
)


# -------------------------------------------------
# Task Configuration
# -------------------------------------------------

# Initial Recommendation Generation Task
data_analysis_task = Task(
    description=(
        "Use the KNN model to generate initial recommendations for user {user_id} from the dataset {csv_path}."
    ),
    expected_output="A dictionary containing user data and a list of initial recommendations.",
    agent=data_analyst
)

# Recommendation Enrichment Task
recommendation_task_llm = Task(
    description=(
        "Enrich the initial recommendations using the LLM, considering current fashion trends and the season {current_season}."
    ),
    expected_output="A list of detailed and enriched recommendations for the user.",
    agent=recommendation_generator_llm,
    inputs={
        "current_season": "current_season",
        "user_data": "data_analysis_task.output.user_data",
        "recommendations": "data_analysis_task.output.recommendations"
    }
)


# -------------------------------------------------
# Crew Configuration
# -------------------------------------------------
crew_llm = Crew(
    agents=[data_analyst, recommendation_generator_llm],
    tasks=[data_analysis_task, recommendation_task_llm],
    process=Process.sequential
)


# -------------------------------------------------
# Execution with Streamlit
# -------------------------------------------------
if __name__ == "__main__":
    st.title("Advanced Fashion Recommendation System with LLM")
    st.sidebar.header("Parameters")

    # User Input
    user_id = st.sidebar.text_input("Enter User ID:")
    csv_path = st.sidebar.text_input("Path to CSV file:", value="./fsData06t.csv")

    # Get current season - north hemisphere
    
    current_month = datetime.datetime.now().month

    if current_month in [12, 1, 2]:
        current_season = "winter"
    elif current_month in [3, 4, 5]:
        current_season = "spring"
    elif current_month in [6, 7, 8]:
        current_season = "summer"
    else:
        current_season = "autumn"

    # Button to execute the system
    if st.sidebar.button("Generate Advanced Recommendations"):
        try:
            # Logs and Report
            st.subheader("Processing Log")
            with st.spinner("Generating recommendations..."):
                results = crew_llm.kickoff(inputs={"csv_path": csv_path, "user_id": user_id, "current_season": current_season})
            
            # Display recommendations
            st.subheader("Personalized Recommendations")
            st.markdown(results)

            st.success("Process completed successfully!")

        except Exception as e:
            st.error(f"Error during processing: {e}")
