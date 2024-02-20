import pickle
import pandas as pd
import streamlit as st

# Load the pre-trained model
merged_df=pd.read_csv("merged_df.csv")
pickle_in = open("model.pkl", "rb")
model = pickle.load(pickle_in)

def predict_BRS(user_id, n, all_books):
    user_books = all_books[all_books['user_id'] == user_id]['book_title'].unique()

    # Remove books already rated by the user
    to_predict = [book for book in all_books['book_title'].unique() if book not in user_books]

    # Make predictions for the books to predict
    test_data = [(user_id, book, 0) for book in to_predict]
    predictions = model.test(test_data)

    # Sort the predictions and get the top N recommendations
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # Get book titles for the top N recommendations
    recommended_books = [item[1] for item in top_n]

    return recommended_books

def main():
    st.set_page_config(page_title="Book Recommendation App", page_icon="📚", layout="wide")

    st.title("Book Recommendations")
    st.markdown(
        """
        <div style="background-color: royalblue; padding: 10px">
            <h2 style="color: white; text-align: center;">SmartRead Picks</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for user input
    st.sidebar.header("User Input")
    user_id = st.sidebar.text_input("User_ID", "Enter User_ID")
    Number = st.sidebar.text_input("Number of Books to Recommend", "5")

    # Convert inputs to integers
    user_id = int(user_id) if user_id.isdigit() else None
    Number = int(Number) if Number.isdigit() else None

    if st.sidebar.button("Recommend") and user_id is not None and Number is not None:
        # Assuming 'final' is your DataFrame
        all_books = merged_df[['user_id', 'book_title']]
        recommendations = predict_BRS(user_id=user_id, n=Number, all_books=all_books)

        # Display the recommended books in a cleaner layout
        st.subheader("Recommended Books:")
        for i, book in enumerate(recommendations, 1):
            st.write(f"{i}. {book}")

if __name__ == '__main__':
    main()
