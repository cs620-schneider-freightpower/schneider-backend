import pandas as pd
from typing import Dict, List, Tuple

class RecommendationTester:
        def __init__(self, df_original, df_model, loads_df, engine):
            self.df_original = df_original
            self.df_model = df_model
            self.loads_df = loads_df
            self.engine = engine
            self.results = []

        def get_user_history_stats(self, user_id: int) -> Dict:

            # get the info for this user
            user_data = self.df_original[self.df_original["USER_PSEUDO_ID"] == user_id]

            if len(user_data) == 0:
                return None
            
            # get the # times they searched/were in specific cities
            geo_city_counts = user_data['GEO_CITY_STANDARDIZED'].value_counts().to_dict()
            event_origin_counts = user_data['EVENT_ORIGIN'].value_counts().to_dict()
            event_destination_counts = user_data['EVENT_DESTINATION'].value_counts().to_dict()

            return {
            'user_id': user_id,
            'total_searches': len(user_data),
            'geo_city_counts': geo_city_counts,
            'event_origin_counts': event_origin_counts,
            'event_destination_counts': event_destination_counts,
            'top_geo_city': geo_city_counts.get(max(geo_city_counts, key=geo_city_counts.get), None) if geo_city_counts else None,
            'top_origin': event_origin_counts.get(max(event_origin_counts, key=event_origin_counts.get), None) if event_origin_counts else None,
            'top_destination': event_destination_counts.get(max(event_destination_counts, key=event_destination_counts.get), None) if event_destination_counts else None,
            }
        
        # get the pickup/dropoff location of specific load
        def get_location_from_load(self, load: pd.Series) -> Tuple[str, str]:
            pickup_location = f"{load['pickup']['city'].upper()},{load['pickup']['state'].upper()}"
            delivery_location = f"{load['delivery']['city'].upper()},{load['delivery']['state'].upper()}"
            return pickup_location, delivery_location

        # calculate how accurate the match is 
        def calculate_match_score(self, recommendations: List[Dict], user_history: Dict) -> Dict:
            if not recommendations or not user_history:
                return None

            geo_cities = user_history['geo_city_counts']
            top_locations = list(geo_cities.keys())[:5]

            top_states = set()
            for loc in top_locations:
                # if there's a comma in the location it has the state
                if ',' in loc:
                    top_states.add(loc.split(',')[1].strip())

            matches = {
                'exact_location_matches': 0,
                'state_matches': 0,
                'no_matches': 0,
                'matched_load_ids': [],
                'unmatched_load_ids': []
            }

            # go through those top 5 recommendations
            for rec in recommendations:
                load = self.loads_df[self.loads_df['id'] == int(rec['load_id'])].iloc[0]
                pickup_loc, delivery_loc = self.get_location_from_load(load)

                # check for exact location match (pickup or delivery)
                if pickup_loc in top_locations or delivery_loc in top_locations:
                    matches['exact_location_matches'] += 1
                    matches['matched_load_ids'].append(rec['load_id'])
                # check for state match
                else:
                    pickup_state = pickup_loc.split(',')[1].strip() if ',' in pickup_loc else None
                    delivery_state = delivery_loc.split(',')[1].strip() if ',' in delivery_loc else None
                    
                    if (pickup_state in top_states or delivery_state in top_states):
                        matches['state_matches'] += 1
                        matches['matched_load_ids'].append(rec['load_id'])
                    else:
                        matches['no_matches'] += 1
                        matches['unmatched_load_ids'].append(rec['load_id'])

            total = len(recommendations)
            accuracy = (matches['exact_location_matches'] + matches['state_matches']) / total if total > 0 else 0

            return {
                'exact_matches': matches['exact_location_matches'],
                'state_matches': matches['state_matches'],
                'no_matches': matches['no_matches'],
                'accuracy_score': accuracy,
                'matched_load_ids': matches['matched_load_ids'],
                'unmatched_load_ids': matches['unmatched_load_ids'],
                'top_user_locations': top_locations
            }
        
        # run the test for a user
        def test_user(self, user_id: int, current_location: Tuple = None) -> Dict:
            user_history = self.get_user_history_stats(user_id)
            if not user_history:
                print("User not found")
                return None
            
            try:
                recommendations = self.engine.get_recommendations(user_id, current_location = current_location)
            except Exception as e:
                print("Error getting recommendations")
                return None
            
            match_score = self.calculate_match_score(recommendations, user_history)

            result = {
            'user_id': user_id,
            'user_history': user_history,
            'recommendations': recommendations,
            'match_analysis': match_score,
            'current_location': current_location
            }

            self.results.append(result)
            return result
        
        def print_report(self, result: Dict) -> None:
            if not result:
                return
            
            user_id = result['user_id']
            history = result['user_history']
            match = result['match_analysis']
            recs = result['recommendations']

            print(f"\n{'='*80}")
            print(f"TEST REPORT FOR USER {user_id}")
            print(f"{'='*80}\n")

            print(f"SEARCH HISTORY:")
            print(f"  Total Searches: {history['total_searches']}")
            print(f"  Top Geo City: {history['top_geo_city']}")
            print(f"  Top Origin: {history['top_origin']}")
            print(f"  Top Destination: {history['top_destination']}")
            
            print(f"\n  Top 5 Geo Locations:")
            for i, (loc, count) in enumerate(list(history['geo_city_counts'].items())[:5], 1):
                print(f"    {i}. {loc}: {count} searches")

            print(f"\n  Top 5 Origins:")
            for i, (origin, count) in enumerate(list(history['event_origin_counts'].items())[:5], 1):
                print(f"    {i}. {origin}: {count} searches")

            print(f"\n  Top 5 Destinations:")
            for i, (dest, count) in enumerate(list(history['event_destination_counts'].items())[:5], 1):
                print(f"    {i}. {dest}: {count} searches")

            print(f"\n{'='*80}")
            print(f"RECOMMENDATION ACCURACY")
            print(f"{'='*80}\n")
            
            print(f"  Accuracy Score: {match['accuracy_score']:.2%}")
            print(f"  Exact Location Matches: {match['exact_matches']}/5")
            print(f"  State Matches: {match['state_matches']}/5")
            print(f"  No Matches: {match['no_matches']}/5")
            
            print(f"\n  User's Top Locations (from history):")
            for i, loc in enumerate(match['top_user_locations'], 1):
                print(f"    {i}. {loc}")

            print(f"\nRECOMMENDED LOADS:")
            print(f"{'Rank':<5} {'Load ID':<8} {'Score':<10} {'Pickup':<25} {'Delivery':<25} {'Match':<15}")
            print(f"{'-'*90}")
            
            for i, rec in enumerate(recs, 1):
                load = self.loads_df[self.loads_df['id'] == int(rec['load_id'])].iloc[0]
                pickup = f"{load['pickup']['city']},{load['pickup']['state']}"
                delivery = f"{load['delivery']['city']},{load['delivery']['state']}"
                
                if rec['load_id'] in match['matched_load_ids']:
                    match_status = "✓ Matched"
                else:
                    match_status = "✗ No Match"
                
                print(f"{i:<5} {rec['load_id']:<8} {rec['recommendation_score']:<10.4f} {pickup:<25} {delivery:<25} {match_status:<15}")

            print(f"\n{'='*80}\n")

if __name__ == "__main__":
    from recommendation import (
        initialize_engine, 
        normalize_geo_to_event,
        get_prev_searches,
        calculate_load_quality,
        get_latlon,
        fix_coord,
        generate_mock_loads
    )

    # get the arguments for RecommendationTester
    engine = initialize_engine(data_path="click-stream(in).csv", loads_path="mock_loads.json")

    df_original = pd.read_csv("click-stream(in).csv")
    df_original['USER_PSEUDO_ID'] = df_original['USER_PSEUDO_ID'].astype(int)
    df_original = normalize_geo_to_event(df_original)
    df_original = df_original.sort_values(['USER_PSEUDO_ID', 'EVENT_TIMESTAMP']).reset_index(drop=True)
    df_original = get_prev_searches(df_original, n_features=3)


    tester = RecommendationTester(df_original=df_original, df_model=engine.df_model, loads_df=engine.loads_df, engine = engine)
    
    # test the top 10 most appeared users 
    top_10_users = df_original['USER_PSEUDO_ID'].value_counts().head(10).index.tolist()

    for i, user_id in enumerate(top_10_users, 1):
        result = tester.test_user(user_id = user_id)

        if result:
            tester.print_report(result)
        else:
            print("Failed to test")



