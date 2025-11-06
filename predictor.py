import joblib
import os
import math
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import sys

# --- CONFIGURATION & ROBUST PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_STG1 = os.path.join(SCRIPT_DIR, "models_lgbm_tuned")
MODEL_DIR_STG2_PYTORCH = os.path.join(SCRIPT_DIR, "models_refinement_pytorch")
MAX_SEQ_LEN = 5
OUTPUT_SEQ_LEN = 3 # The decoder predicts 3 future waypoints
RNN_HIDDEN_SIZE = 128
NLP_EMBEDDING_SIZE = 384
SIGHTING_FEATURE_SIZE = 3 + NLP_EMBEDDING_SIZE # lat/lon/time + embedding
STATIC_FEATURE_SIZE = 2 # risk_level, dist_to_nearest_city

# --- PYTORCH MODEL DEFINITIONS (Seq2Seq Refinement Engine) ---

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Input size: Sighting features (387)
        self.lstm = nn.LSTM(input_size=SIGHTING_FEATURE_SIZE, hidden_size=RNN_HIDDEN_SIZE, batch_first=True)
        
    def forward(self, seq_input):
        # lstm_out: output features (h_t) from all timesteps
        # hidden: (h_n, c_n) the hidden state and cell state of the last timestep
        _, hidden = self.lstm(seq_input)
        return hidden

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Input to the LSTM at each step is the predicted coordinate from the previous step (2) 
        # plus the static features (2)
        # Input size: 2 (coords) + 2 (static) = 4
        self.lstm = nn.LSTM(input_size=4, hidden_size=RNN_HIDDEN_SIZE, batch_first=True)
        self.fc_out = nn.Linear(RNN_HIDDEN_SIZE, 2) # Predicts 2 coordinates (lat, lon)

    def forward(self, static_input, initial_hidden):
        batch_size = static_input.size(0)
        
        # 1. Prepare initial input: use the last sighting (or zero padding if none exist)
        # In a full production Seq2Seq setup, we would use the last sighting here.
        # For simplicity and training compatibility, we use a zero vector for the first step, 
        # relying on the hidden state for context.
        decoder_input = torch.zeros(batch_size, 1, 2).to(static_input.device) 
        
        # Static features are concatenated to the decoder input at every step
        static_repeated = static_input.unsqueeze(1).repeat(1, OUTPUT_SEQ_LEN, 1)

        # 2. Iterate through the sequence length (predict N future points)
        hidden = initial_hidden
        outputs = []
        
        for i in range(OUTPUT_SEQ_LEN):
            # Concatenate previous output (or zero) with static features
            # The input for this step is [2 (from last step) + 2 (static features)]
            current_lstm_input = torch.cat((decoder_input, static_input.unsqueeze(1)), dim=-1)
            
            lstm_out, hidden = self.lstm(current_lstm_input, hidden)
            
            # Predict the next coordinate (2 dimensions)
            predicted_coord = self.fc_out(lstm_out.squeeze(1))
            outputs.append(predicted_coord)
            
            # Use the predicted coordinates as the input for the next step
            decoder_input = predicted_coord.unsqueeze(1) 
            
        # Stack the outputs into a sequence (Batch x SeqLen x Coords)
        return torch.stack(outputs, dim=1)

class RefinementEngineSeq2Seq(nn.Module):
    def __init__(self):
        super(RefinementEngineSeq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.dense_out = nn.Linear(RNN_HIDDEN_SIZE + STATIC_FEATURE_SIZE, 2)

    def forward(self, seq_input, static_input):
        # 1. Encode the past sighting sequence
        (h_n, c_n) = self.encoder(seq_input)
        
        # 2. Decode the future path. The decoder's initial state is the encoder's final state.
        predicted_path_sequence = self.decoder(static_input, (h_n, c_n))

        # 3. Final Prediction: Use the last predicted point in the path as the recovery location.
        # Alternatively, for robustness, we could average the sequence. We'll use the last point.
        final_prediction = predicted_path_sequence[:, -1, :]
        
        return final_prediction

# --- GLOBAL MODEL LOADING ---
try:
    print("--- Sachet AI Engine Initializing: Loading all models... ---")
    
    # Load Stage 1 (LightGBM) Models
    PIPELINE_STG1 = joblib.load(os.path.join(MODEL_DIR_STG1, 'pipeline.joblib'))
    CLF_RISK = joblib.load(os.path.join(MODEL_DIR_STG1, 'clf_risk.joblib'))
    CLF_RECOVERED = joblib.load(os.path.join(MODEL_DIR_STG1, 'clf_recovered.joblib'))
    REG_TIME = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_time.joblib'))
    REG_LAT = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_lat.joblib'))
    REG_LON = joblib.load(os.path.join(MODEL_DIR_STG1, 'reg_recovery_lon.joblib'))
    print("Stage 1 (LightGBM) Models loaded successfully.")

    # Load Stage 2 (PyTorch Trajectory) Models
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use the new Seq2Seq model class
    REFINEMENT_MODEL = RefinementEngineSeq2Seq().to(DEVICE)
    REFINEMENT_MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR_STG2_PYTORCH, 'refinement_model.pth'), map_location=torch.device(DEVICE)))
    REFINEMENT_MODEL.eval() 
    print("Stage 2 (PyTorch) Seq2Seq Refinement Engine loaded successfully.")
    
    # Load NLP Model
    NLP_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    print("\n--- All prediction models loaded successfully. AI Engine is ready. ---")

except FileNotFoundError as e:
    # Error handling remains the same
    print("\n" + "="*80)
    print("FATAL ERROR: A required model file was not found.")
    print(f"Missing File Details: {e}")
    print("="*80 + "\n")
    sys.exit(1) 

# --- HELPER FUNCTIONS (Rest remain the same) ---
def get_feature_names_compat(enc, input_features):
    if hasattr(enc, "get_feature_names_out"): return list(enc.get_feature_names_out(input_features));
    names = [];
    for feat, cats in zip(input_features, enc.categories_):
        for c in cats: names.append(f"{feat}_{c}")
    return names

def haversine(lat1, lon1, lat2, lon2):
    """The definitive, fully vectorized haversine distance function."""
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians([lat1, lon1, lat2, lon2])
    d_lon = lon2_rad - lon1_rad; d_lat = lat2_rad - lat1_rad
    a = np.sin(d_lat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)); return R * c

def prepare_input_stg1(inp: dict, pipeline: dict):
    """Prepares user input for the Stage 1 LightGBM models."""
    df=pd.DataFrame([inp])
    df['hour_sin']=np.sin(2 * math.pi * df['abduction_time'] / 24.0); df['hour_cos']=np.cos(2 * math.pi * df['abduction_time'] / 24.0)
    CITY_CENTERS={"Mumbai":(19.0761, 72.8775),"Pune":(18.5203, 73.8567),"Nagpur":(21.1497, 79.0806),"Nashik":(19.9975, 73.7898)}
    df['dist_to_nearest_city']=df.apply(lambda row:min([haversine(row['latitude'],row['longitude'],c_lat,c_lon) for c_lat,c_lon in CITY_CENTERS.values()]), axis=1)
    X_num_scaled=pipeline['scaler'].transform(df[pipeline['num_cols']])
    X_cat_encoded=pipeline['encoder'].transform(df[pipeline['cat_cols']])
    cat_feature_names=get_feature_names_compat(pipeline['encoder'], pipeline['cat_cols'])
    X_final=pd.concat([pd.DataFrame(X_num_scaled,columns=pipeline['num_cols']),pd.DataFrame(X_cat_encoded,columns=cat_feature_names)], axis=1)
    return X_final[pipeline['X_columns']]

# --- MAIN PREDICTION FUNCTIONS (Called by the App) ---
def predict_initial_case(inp: dict):
    X_in=prepare_input_stg1(inp, PIPELINE_STG1); risk_label=int(CLF_RISK.predict(X_in)[0]); risk_prob=float(CLF_RISK.predict_proba(X_in)[0].max()); rec_prob=float(CLF_RECOVERED.predict_proba(X_in)[0][1]); rec_label=1 if rec_prob>=0.5 else 0
    est_recovery_time, pred_lat, pred_lon = 0.0, 0.0, 0.0
    if rec_label == 1:
        est_recovery_time=float(REG_TIME.predict(X_in)[0]); pred_lat=float(REG_LAT.predict(X_in)[0]); pred_lon=float(REG_LON.predict(X_in)[0])
    return {'risk_label':risk_label,'risk_prob':risk_prob,'recovered_label':rec_label,'recovered_prob':rec_prob,'recovery_time_hours':est_recovery_time,'predicted_latitude':pred_lat,'predicted_longitude':pred_lon}

def refine_location_with_sightings(initial_prediction: dict, sightings: list, initial_case_input: dict):
    if not sightings or REFINEMENT_MODEL is None:
        return initial_prediction['predicted_latitude'], initial_prediction['predicted_longitude']
    
    # Static features: Risk level and Distance to nearest city
    static_features=torch.tensor([[initial_prediction['risk_label'], initial_case_input['dist_to_nearest_city']]], dtype=torch.float32).to(DEVICE)
    seq_features=[]
    
    # 1. Prepare sequence features
    for sighting in sorted(sightings, key=lambda s: s['hours_since']):
        text_embedding=NLP_MODEL.encode(sighting['direction_text'], device=DEVICE); 
        features=[sighting['lat'],sighting['lon'],sighting['hours_since']]+list(text_embedding); 
        seq_features.append(features)
    
    # 2. Pad/Truncate the sequence (MAX_SEQ_LEN is 5)
    padded_seq=np.zeros((MAX_SEQ_LEN, SIGHTING_FEATURE_SIZE), dtype=np.float32); 
    seq_len = len(seq_features)
    
    if seq_len > 0:
        # Use only the last MAX_SEQ_LEN points
        seq_to_use = seq_features[-MAX_SEQ_LEN:]
        padded_seq[-len(seq_to_use):] = np.array(seq_to_use, dtype=np.float32)

    seq_tensor=torch.tensor([padded_seq], dtype=torch.float32).to(DEVICE)
    
    # 3. Run Seq2Seq Refinement
    with torch.no_grad():
        # REFINEMENT_MODEL outputs the final predicted coordinate (Batch x 2)
        refined_coords = REFINEMENT_MODEL(seq_tensor, static_features).cpu().numpy()[0]
        
    return float(refined_coords[0]), float(refined_coords[1])
