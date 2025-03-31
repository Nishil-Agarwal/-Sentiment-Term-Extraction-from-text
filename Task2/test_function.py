import json
import string
import json
import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from tqdm import tqdm



def clean_text(text):
    for ch in [",", ".", "!", "?", ";", ":", "\"", "'", "(", ")", "_", "-", "/", "\\"]:
        text = text.replace(ch, "")
    return text

def simple_tokenize(text):
    cleaned = clean_text(text)
    return cleaned.split()

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],   
                                   dp[i][j - 1],   
                                   dp[i - 1][j - 1]) 
    return dp[m][n]

def token_matches(token, aspect):
    
    token_clean = clean_text(token).lower()
    aspect_clean = clean_text(aspect).lower()
    
    if token_clean == aspect_clean:
        return True
    
    if token_clean.endswith('s') and token_clean[:-1] == aspect_clean:
        return True

    if aspect_clean in token_clean:
        return True
    
    # 3. Fuzzy matching using Levenshtein distance.
    threshold = max(1, int(0.2 * len(aspect_clean)))  # Allow 20% differences.
    distance = levenshtein_distance(token_clean, aspect_clean)
    if distance <= threshold:
        return True

    
    return False


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)
            aspect_input_ids = batch["aspect_input_ids"].to(device)
            aspect_attention_mask = batch["aspect_attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(context_input_ids, context_attention_mask,
                           aspect_input_ids, aspect_attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def preprocess_file(input_path, output_path):
    with open(input_path, 'r') as infile:
        data = json.load(infile)
    
    new_data = []
    
    for item in data:
        sentence = item['sentence']
        tokens = simple_tokenize(sentence)
        
        
        for aspect in item.get('aspect_terms', []):
            term = aspect.get('term', "")
            polarity = aspect.get('polarity', "")
            
            term_tokens = simple_tokenize(term)
            aspect_word = term_tokens[0] if term_tokens else term
            
            index = -1
            for i, token in enumerate(tokens):
                if token_matches(token, aspect_word):
                    index = i
                    break
            
            new_instance = {
                "tokens": tokens,
                "polarity": polarity,
                "aspectterm": [term],
                "index": index
            }
            new_data.append(new_instance)
    
    with open(output_path, 'w') as outfile:
        json.dump(new_data, outfile, indent=4)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

senitment_mapping = {"negative": 0, "neutral": 1, "positive": 2}

class ABSADataset(Dataset):
    def __init__(self, json_path, tokenizer, max_context_len=128, max_aspect_len=16):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = data
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_aspect_len = max_aspect_len

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = " ".join(sample["tokens"])
        if isinstance(sample["aspectterm"], list):
            aspect = " ".join(sample["aspectterm"])
        else:
            aspect = sample["aspectterm"]
        # aspect term is retrieved from the .json not from the tokenized sample itself 
        polarity_str = sample["polarity"]
        label = senitment_mapping.get(polarity_str.lower(), 1)  # create mapping
        
        context_enc = self.tokenizer.encode_plus(    #using bert
            context,
            add_special_tokens=True,
            max_length=self.max_context_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        aspect_enc = self.tokenizer.encode_plus(     #using bert 
            aspect,
            add_special_tokens=True,
            max_length=self.max_aspect_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        context_input_ids = context_enc['input_ids'].squeeze(0)
        context_attention_mask = context_enc['attention_mask'].squeeze(0)
        aspect_input_ids = aspect_enc['input_ids'].squeeze(0)
        aspect_attention_mask = aspect_enc['attention_mask'].squeeze(0)
        
        return {
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "aspect_input_ids": aspect_input_ids,
            "aspect_attention_mask": aspect_attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

        
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, hidden_states, guide_vector):
        guide = guide_vector.unsqueeze(1)
        projected = self.tanh(self.linear(hidden_states))
        scores = (projected * guide).sum(dim=2)
        weights = self.softmax(scores)
        weighted_sum = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return weighted_sum, weights
    
    


class IAN_Model(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_labels, dropout=0.5, cell_type="gru"):
        super(IAN_Model, self).__init__()
        self.bert = bert_model
        self.bert_hidden_size = bert_model.config.hidden_size
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.lower()
        
        if self.cell_type == "gru":
            self.rnn_context = nn.GRU(input_size=self.bert_hidden_size, hidden_size=hidden_dim,
                                      num_layers=1, batch_first=True, bidirectional=True)              # bidirectional encodings 
            self.rnn_aspect = nn.GRU(input_size=self.bert_hidden_size, hidden_size=hidden_dim,
                                     num_layers=1, batch_first=True, bidirectional=True)               # bidirectional encodings
        elif self.cell_type == "lstm":
            self.rnn_context = nn.LSTM(input_size=self.bert_hidden_size, hidden_size=hidden_dim,
                                       num_layers=1, batch_first=True, bidirectional=True)
            self.rnn_aspect = nn.LSTM(input_size=self.bert_hidden_size, hidden_size=hidden_dim,
                                      num_layers=1, batch_first=True, bidirectional=True)
        else:
            raise ValueError("Unsupported cell type. Choose 'gru' or 'lstm'.")
        
        self.attention_context = Attention(hidden_dim * 2)
        self.attention_aspect = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_labels)
    
    def forward(self, context_input_ids, context_attention_mask, 
                aspect_input_ids, aspect_attention_mask):
        context_outputs = self.bert(input_ids=context_input_ids, attention_mask=context_attention_mask)
        aspect_outputs = self.bert(input_ids=aspect_input_ids, attention_mask=aspect_attention_mask)
        context_embeds = context_outputs.last_hidden_state
        aspect_embeds = aspect_outputs.last_hidden_state
        
        if self.cell_type == "gru":
            context_rnn, _ = self.rnn_context(context_embeds)
            aspect_rnn, _ = self.rnn_aspect(aspect_embeds)
        else: 
            context_rnn, _ = self.rnn_context(context_embeds)
            aspect_rnn, _ = self.rnn_aspect(aspect_embeds)
        
        context_mask = context_attention_mask.unsqueeze(-1).float()
        aspect_mask = aspect_attention_mask.unsqueeze(-1).float()
        context_sum = torch.sum(context_rnn * context_mask, dim=1)
        aspect_sum = torch.sum(aspect_rnn * aspect_mask, dim=1)
        context_lengths = torch.clamp(torch.sum(context_mask, dim=1), min=1e-9)    # clamps all elements in input into the range [ min , max ] 
        aspect_lengths = torch.clamp(torch.sum(aspect_mask, dim=1), min=1e-9)
        context_pool = context_sum / context_lengths
        aspect_pool = aspect_sum / aspect_lengths
        
        context_final, _ = self.attention_context(context_rnn, aspect_pool)
        aspect_final, _ = self.attention_aspect(aspect_rnn, context_pool)
        final_rep = torch.cat((context_final, aspect_final), dim=1)
        final_rep = self.dropout(final_rep)
        logits = self.fc(final_rep)
        return logits


def test_function(model_path, test_json_path):
    preprocess_file(test_json_path , 'test_tokenized.json')
    test_tokenized_path = 'test_tokenized.json'
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    
    hidden_dim = 128
    num_labels = 3
    cell_type = "gru" 
    model = IAN_Model(bert_model, hidden_dim, num_labels, dropout=0.5, cell_type=cell_type)   #The implementation of IAN_Model is given above
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    

    test_dataset = ABSADataset(test_tokenized_path , tokenizer)             #Its implementations should be run before executing this cell 
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_acc

test_accuracy = test_function("best_model.pt", "test.json")
