from flask import Flask, request, jsonify, make_response
from sqlalchemy import create_engine
from flask_cors import CORS
import pandas as pd
import joblib  # For loading the trained model
import os

# Set up your Flask app
app = Flask(__name__)
CORS(app)


# Database configuration
db_user = 'drewf'    
db_password = 'Soccer.666'  
db_host = 'localhost'      
db_port = '5432'           
db_name = 'drewf'    

# Connection string
connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

# Load the ML model (ensure 'LogisticRegressionModel.pkl' is in the same directory or specify path)
model = joblib.load('ml_model/RandomForest_Model.pkl')

# Feature extraction function
def get_team_features(team, opponent, prefix, odds, match_stats_df, team_ranks_df, team_stats_df):
    features = {}

    team_stats = team_stats_df[team_stats_df['team_name'] == team]
    match_stats = match_stats_df[match_stats_df['team_name'] == team]
    last_three_matches = match_stats[match_stats['match_type'] == 'last_3_matches']
    all_matches = match_stats[match_stats['match_type'] == 'all_matches']
    last_three_pts_avg = last_three_matches['points'].mean()
    last_three_goals_avg = last_three_matches['goals_for'].mean()
    last_three_ga_avg = last_three_matches['goals_against'].mean()

    weights = [0.5, 0.3, 0.2]

    weighted_avg_points = (last_three_matches['points'].values * weights).sum()
    weighted_avg_goals = (last_three_matches['goals_for'].values * weights).sum()
    weighted_avg_goals_against = (last_three_matches['goals_against'].values * weights).sum()

    win_count = 0
    total_matches = 0

    # Iterate through Arsenal's matches
        # Get the current opponent's rank
    opponent_rank = int(team_ranks_df[team_ranks_df['team_name'] == opponent]['rank'].iloc[0])
    
    # Find matches against teams within 2 spots of the opponent's rank
    similar_ranked_opponents = all_matches[
        (all_matches['opponent_rank'] >= opponent_rank - 2) &
        (all_matches['opponent_rank'] <= opponent_rank + 2)
    ]
    
    # Count wins against similar ranked teams
    win_count += similar_ranked_opponents[similar_ranked_opponents['points'] == 3].shape[0]
    goals_count = similar_ranked_opponents['goals_for'].sum()
    ga_count = similar_ranked_opponents['goals_against'].sum()
    total_matches += similar_ranked_opponents.shape[0]

    # Calculate win ratio
    win_ratio = win_count / total_matches if total_matches > 0 else 0
    goals_per_similar_rank = goals_count / total_matches if total_matches > 0 else 0
    ga_per_similar_rank = ga_count / total_matches if total_matches > 0 else 0

    goals_to_ga_ratio = goals_per_similar_rank / ga_per_similar_rank if ga_per_similar_rank > 0 else goals_per_similar_rank


    features[prefix + "odds"] = int(odds)
    features[prefix + "points"] = int(team_stats['points'].iloc[0])
    features[prefix + "rank"] = int(team_ranks_df[team_ranks_df['team_name'] == team]['rank'].iloc[0])
    features[prefix + "goals_for"] = int(team_stats_df['goals_for'].iloc[0])
    features[prefix + 'goals_against'] = int(team_stats_df['goals_against'].iloc[0])
    features[prefix + 'wins'] = int(team_stats_df['wins'].iloc[0])
    features[prefix + 'draws'] = int(team_stats_df['draws'].iloc[0])
    features[prefix + 'losses'] = int(team_stats_df['losses'].iloc[0])
    features[prefix + 'win_streak'] = int(team_stats_df['win_streak'].iloc[0])
    features[prefix + 'loss_streak'] = int(team_stats_df['loss_streak'].iloc[0])
    features[prefix + 'draw_streak'] = int(team_stats_df['draw_streak'].iloc[0])
    features[prefix + 'last_3_avg_pts'] = float(last_three_pts_avg)
    features[prefix + 'last_3_goals'] = float(last_three_goals_avg)
    features[prefix + 'last_3_goals_against'] = float(last_three_ga_avg)
    features[prefix + 'last_3_wavg_pts'] = float(weighted_avg_points)
    features[prefix + 'last_3_wavg_goals'] = float(weighted_avg_goals)
    features[prefix + 'last_3_wavg_goals_against'] = float(weighted_avg_goals_against)
    if prefix == "home_":
        oppon_pts = int(team_stats_df[team_stats_df['team_name'] == opponent]['points'].iloc[0])
        home_away_pts_interaction = int(team_stats['points'].iloc[0]) / oppon_pts if oppon_pts > 0 else int(team_stats['points'].iloc[0])
        features[prefix + 'away_points_interaction'] = float(home_away_pts_interaction)
    features[prefix + 'similar_rank_win_ratio'] = float(win_ratio)
    features[prefix + 'similar_rank_goals'] = float(goals_per_similar_rank)
    features[prefix + 'similar_rank_ga'] = float(ga_per_similar_rank)
    features[prefix + 'similar_rank_goal_ratio'] = float(goals_to_ga_ratio)

    return features

# Flask route for prediction
@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        response = make_response(jsonify({'message': 'CORS preflight response'}), 200)
        response.headers.add('Access-Control-Allow-Origin', '*')  # Or specify your React app's origin
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    # Get user input from request
    data = request.json
    home_team, away_team, home_odds, away_odds, draw_odds = data.get('homeTeam'), data.get('awayTeam'), data.get('homeOdds'), data.get('awayOdds'), data.get('drawOdds')

    # Query the database for match and team data
    try:
        match_stats_query = "SELECT * FROM matches WHERE team_name = %s OR team_name = %s"
        team_stats_query = "SELECT * FROM teams WHERE team_name = %s OR team_name = %s"
        ranks_query = "SELECT * FROM team_rankings"

        match_stats_df = pd.read_sql_query(match_stats_query, engine, params=(home_team, away_team))
        team_stats_df = pd.read_sql_query(team_stats_query, engine, params=(home_team, away_team))
        team_ranks_df = pd.read_sql_query(ranks_query, engine)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Generate features for both teams
    home_features = get_team_features(home_team, away_team, 'home_', home_odds, match_stats_df, team_ranks_df, team_stats_df)
    away_features = get_team_features(away_team, home_team, 'away_', away_odds, match_stats_df, team_ranks_df, team_stats_df)

    # # Combine all features and predict
    # all_features = {**home_features, **away_features}
    # all_features['draw_odds'] = int(draw_odds)
    # feature_df = pd.DataFrame([all_features])
    # prediction = model.predict(feature_df)
    # predicted_winner = home_team if prediction[0] == 2 else away_team if prediction[0] == 1 else 'Draw'

    # return jsonify({'prediction': predicted_winner})
    # Combine all features and predict
    all_features = {**home_features, **away_features}
    all_features['draw_odds'] = int(draw_odds)
    feature_df = pd.DataFrame([all_features])

    # Get the predicted probabilities
    probabilities = model.predict_proba(feature_df)
    # Map probabilities to their respective outcomes
    prob_home_win = probabilities[0][2] * 100  # Assuming index 2 is home win
    prob_away_win = probabilities[0][1] * 100  # Assuming index 1 is away win
    prob_draw = probabilities[0][0] * 100       # Assuming index 0 is draw
    
    # Determine the predicted winner based on probabilities
    predicted_winner = home_team if probabilities[0][2] > max(probabilities[0][1], probabilities[0][0]) else \
                       away_team if probabilities[0][1] > max(probabilities[0][2], probabilities[0][0]) else 'Draw'

    return jsonify({
        'prediction': predicted_winner,
        'probabilities': {
            'home_win': round(prob_home_win, 2),
            'away_win': round(prob_away_win, 2),
            'draw': round(prob_draw, 2)
        }
    })

if __name__ == '__main__':
    app.run(debug=True)

