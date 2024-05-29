from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the data
file_path = "/Users/joshgross/Downloads/nba_games.csv"
df = pd.read_csv(file_path, index_col=0)

# Load the spread conversion data
spread_conversion_path = "/Users/joshgross/Downloads/Implied Prob of NBA spreads - Sheet1.csv"
spread_conversion_df = pd.read_csv(spread_conversion_path)

# Ensure correct column names in spread conversion data
spread_conversion_df.columns = [col.strip() for col in spread_conversion_df.columns]

# Display the column names for debugging
print("Spread Conversion Data Columns:", spread_conversion_df.columns)

# Preprocess data
important_data = df[['team', 'home', 'date', 'fga', 'ast', '3pa', 'stl', 'orb', 'ft', 'tov', 'fta', '3p%', 'won', 'team_opp']]
important_data['date'] = pd.to_datetime(important_data['date'], errors='coerce')
important_data = important_data.dropna(subset=['date'])  # Remove rows with invalid dates
important_data = important_data.sort_values(by='date')

# Define important stats
important_stats = ['fga', 'ast', '3pa', 'stl', 'orb', 'ft', 'tov', 'fta', '3p%']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats', methods=['POST'])
def stats():
    team1 = request.form['team1']
    team2 = request.form.get('team2')
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except Exception as e:
        return f"Invalid date format: {e}"

    if start_date > end_date:
        return "Start date must be before end date."

    # Filter data within the date range for team 1
    team1_data = important_data[(important_data['team'] == team1) & 
                                (important_data['date'] >= start_date) & 
                                (important_data['date'] <= end_date)]
    
    if team1_data.empty:
        return f"No data available for team {team1} between {start_date.date()} and {end_date.date()}."
    
    team1_averages = team1_data[important_stats].mean()

    # Initialize the response with team 1 stats
    response = f"<h2>Running Averages for {team1} from {start_date.date()} to {end_date.date()}:</h2>" + team1_averages.to_frame().to_html()

    # If a second team is provided, calculate its stats and add to the response
    if team2:
        team2_data = important_data[(important_data['team'] == team2) & 
                                    (important_data['date'] >= start_date) & 
                                    (important_data['date'] <= end_date)]
        
        if team2_data.empty:
            response += f"<br><br>No data available for team {team2} between {start_date.date()} and {end_date.date()}."
        else:
            team2_averages = team2_data[important_stats].mean()
            response += f"<br><br><h2>Running Averages for {team2} from {start_date.date()} to {end_date.date()}:</h2>" + team2_averages.to_frame().to_html()

            # Combine team1 and team2 stats for prediction
            combined_stats = pd.DataFrame([team1_averages, team2_averages])
            combined_stats.reset_index(drop=True, inplace=True)
            
            # Load and preprocess the data for training
            X = important_data[important_stats]
            y = important_data['won']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the logistic regression model
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            # Scale the combined stats
            combined_stats_scaled = scaler.transform(combined_stats)

            # Predict the probabilities
            predictions = model.predict_proba(combined_stats_scaled)[:, 1]

            response += f"<br><br><h2>Prediction Probabilities:</h2>"
            response += f"<p>Probability of {team1} winning: {predictions[0]:.2f}</p>"
            response += f"<p>Probability of {team2} winning: {predictions[1]:.2f}</p>"

    return response

if __name__ == '__main__':
    app.run(debug=True)