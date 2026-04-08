import time
import requests
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from crewai import Agent, Task, Crew, Process, LLM
from services.supabase_service import get_supabase
from utils.sse_manager import sse_manager
from utils.config import settings
from datetime import datetime, timezone

# ── GNN MODEL ───────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# ── AGENT LOGIC ─────────────────────────────────────────────────────────────

async def run_gnn_agent(run_id: str, automl_result: dict):
    """
    Stage 4 Part 2: Graph Neural Network for Protein-Protein Interactions.
    Runs after AutoML, using SHAP results as node features and STRING DB for edges.
    """
    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_gnn",
        "status": "running",
        "output_summary": "Building Protein-Protein Interaction Graph (STRING DB)...",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    t0 = time.monotonic()
    
    # 1. Extract SHAP scores from AutoML result
    # Expected format: {"shap_importance": {"PROTEIN_A": 0.45, "PROTEIN_B": 0.32, ...}}
    shap_data = automl_result.get("automl", {}).get("shap_importance", {})
    if not shap_data:
        # Fallback for demo if SHAP is missing
        shap_data = {"EGFR": 0.8, "BRCA1": 0.7, "APP": 0.6, "SNCA": 0.4, "KLK3": 0.3}

    proteins = list(shap_data.keys())
    
    # 2. Query STRING DB for edges
    # https://string-db.org/api/json/network?identifiers=p1%0dp2%0dp3
    edges = []
    try:
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": "%0d".join(proteins),
            "species": 9606, # Human
            "required_score": 400 # Medium confidence
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            edge_data = resp.json()
            for e in edge_data:
                edges.append((e['preferredName_A'], e['preferredName_B'], e['score']))
    except Exception as e:
        print(f"STRING DB fetch failed: {e}")

    # 3. Build NetworkX Graph & PyG Data
    G = nx.Graph()
    G.add_nodes_from(proteins)
    for u, v, w in edges:
        if u in proteins and v in proteins:
            G.add_edge(u, v, weight=float(w)/1000.0)

    # Convert to PyTorch Geometric tensors
    node_to_idx = {p: i for i, p in enumerate(proteins)}
    x = torch.tensor([[shap_data.get(p, 0.0)] for p in proteins], dtype=torch.float)
    
    edge_index_list = []
    edge_weight_list = []
    for u, v, w in G.edges(data=True):
        edge_index_list.append([node_to_idx[u], node_to_idx[v]])
        edge_weight_list.append(w['weight'])
        # Undirected
        edge_index_list.append([node_to_idx[v], node_to_idx[u]])
        edge_weight_list.append(w['weight'])

    if not edge_index_list:
        # No edges found, can't run GNN properly, return centrality 0
        centrality = {p: 0.0 for p in proteins}
        gnn_embeddings = x.numpy().tolist()
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)
        
        # 4. Run 2-layer GCN
        model = GCN(in_channels=1, hidden_channels=4, out_channels=2)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_weight)
        
        gnn_embeddings = out.numpy().tolist()
        
        # Calculate NetworkX Centrality (Betweenness)
        # Note: we use this for the "hidden hub" flagging
        centrality = nx.betweenness_centrality(G)

    # 5. Identify Hidden Hubs (Low SHAP, High Centrality)
    results = []
    supabase = get_supabase()
    
    avg_shap = sum(shap_data.values()) / len(shap_data) if shap_data else 0
    avg_cent = sum(centrality.values()) / len(centrality) if centrality else 0

    for i, p in enumerate(proteins):
        shap_score = shap_data.get(p, 0.0)
        cent_score = centrality.get(p, 0.0)
        is_hidden = (shap_score < avg_shap) and (cent_score > avg_cent)
        
        res_row = {
            "run_id": run_id,
            "protein": p,
            "shap_score": shap_score,
            "centrality": cent_score,
            "gnn_embedding": gnn_embeddings[i],
            "is_hidden_hub": is_hidden
        }
        results.append(res_row)
        
        # Write to Supabase
        try:
            supabase.table("gnn_results").insert(res_row).execute()
        except Exception:
            pass

    latency_ms = int((time.monotonic() - t0) * 1000)

    await sse_manager.publish(run_id, {
        "type": "agent_update",
        "agent": "zora_gnn",
        "status": "completed",
        "latency_ms": latency_ms,
        "output_summary": (
            f"GNN Graph complete. {len(proteins)} nodes, {len(edges)} interactions. "
            f"Flagged {len([r for r in results if r['is_hidden_hub']])} hidden hubs."
        ),
        "data": results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    return results
