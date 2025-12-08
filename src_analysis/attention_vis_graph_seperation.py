# %%
import yaml
from pathlib import Path
import networkx as nx
import numpy as np
from easydict import EasyDict as edict
import torch
import lightning as L
import torch_geometric.data
from torch_geometric.utils import to_networkx
import matplotlib
from matplotlib import pyplot as plt

import src_analysis.random_walks as random_walks  # pylint: disable=import-error

from src.data import DatasetBuilder
from src.data import GraphSeparationCSLDataset, GraphSeparationCSLWalker
from src.data import GraphSeparationSR16Dataset, GraphSeparationSR16Walker
from src.data import GraphSeparationSR25Dataset, GraphSeparationSR25Walker
from src.model import Model
from src.train.lit_module import LitModule

dataset_name = 'SR25'

# %%
args = edict()
if dataset_name == 'SR16':
    args.config = 'configs/graph_separation/sr16_deberta.yaml'
elif dataset_name == 'SR25':
    args.config = 'configs/graph_separation/sr25_deberta.yaml'
elif dataset_name == 'CSL':
    args.config = 'configs/graph_separation/csl_deberta.yaml'
with open(args.config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    config = edict(config)
config.test_mode = True
config.test_n_walks = 1
config.compile = False

# %%
L.seed_everything(config.seed, workers=True)
torch.set_float32_matmul_precision('medium')

if dataset_name == 'SR16':
    walker = GraphSeparationSR16Walker(config)
    ds_builder = DatasetBuilder('graph_separation_sr16', True, GraphSeparationSR16Dataset, 'experiments/data', config)
elif dataset_name == 'SR25':
    walker = GraphSeparationSR25Walker(config)
    ds_builder = DatasetBuilder('graph_separation_sr25', True, GraphSeparationSR25Dataset, 'experiments/data', config)
elif dataset_name == 'CSL':
    walker = GraphSeparationCSLWalker(config)
    ds_builder = DatasetBuilder('graph_separation_csl', True, GraphSeparationCSLDataset, 'experiments/data', config)
walker.register_ds_builder(ds_builder)
print(walker)

dataset = ds_builder.train_dataset()
dataset.cuda()
print(dataset)

# %%
model = Model(
    walker=walker,
    backbone='microsoft/deberta-base',
    dropout=None,
    att_dropout=None,
    head_dropout=0.0,
    vocab_size=-1,
    max_length=512,
    pretrained=True,
    pretrained_tokenizer=True,
    is_compiled=False,
    deberta_use_pooler=False,
    debug_mode=False
)
model = LitModule(
    model=model,
    optimizer_config=None,
    lr_scheduler_config=None
)
if dataset_name == 'SR16':
    ckpt_path = 'experiments/checkpoints/graph_separation_sr16/microsoft_deberta-base,pt_True,l_512,w_min_degree,wl_1000,n_True,nw_1,enw_1,b_256,es_validation_loss_min_100,lr_2e-05_2e-05,steps_100000,wu_5000,wd_0.01,clip_2,seed_42,/best.ckpt'
elif dataset_name == 'SR25':
    ckpt_path = 'experiments/checkpoints/graph_separation_sr25/microsoft_deberta-base,pt_True,l_512,w_min_degree,wl_1000,n_True,nw_1,enw_1,b_256x8,es_validation_loss_min_100,lr_2e-05_2e-05,steps_100000,wu_5000,wd_0.01,seed_42,/best.ckpt'
elif dataset_name == 'CSL':
    ckpt_path = 'experiments/checkpoints/graph_separation_csl/microsoft_deberta-base,pt_True,l_512,w_min_degree,wl_1000,n_True,nw_2,enw_8,b_128,es_validation_loss_min_100,lr_2e-05_2e-05,steps_100000,wu_5000,wd_0.01,clip_2,seed_42,/best.ckpt'
checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()

# %%
def draw(root, data_idx, layer_idx, head_idx, seed, draw_graph=False):
    num_layers = 12
    num_heads = 12
    batch_idx = 0

    if seed is not None:
        L.seed_everything(seed, workers=True)

    with torch.autocast(device_type='cuda', dtype=torch.float32):
        batch = torch_geometric.data.Batch.from_data_list([dataset[data_idx]])
        target, n_targets, target_ids = model.model.walker.parse_target(batch)
        n_walks = model.model.walker.eval_n_walks
        start_nodes, target_ids = model.model.walker.get_start_nodes(n_targets, target_ids, n_walks)
        G = nx.to_undirected(to_networkx(batch))
        walks, restarts = random_walks.random_walks(
            G=G,
            n_walks=1,
            walk_len=model.model.walker.walk_length,
            min_degree=model.model.walker.min_degree,
            sub_sampling=model.model.walker.sub_sampling,
            p=model.model.walker.p,
            q=model.model.walker.q,
            alpha=model.model.walker.alpha,
            k=model.model.walker.k,
            no_backtrack=model.model.walker.no_backtrack,
            start_nodes=start_nodes,
            seed=None,
            verbose=False
        )
        if model.model.walker.neighbors:
            A = nx.adjacency_matrix(G)
            indptr = A.indptr.astype(np.uint32)
            indices = A.indices.astype(np.uint32)
            named_walks, walks, restarts, neighbors = random_walks._anonymize_with_neighbors(walks, restarts, indptr, indices)
            text = random_walks._as_text_with_neighbors(named_walks, restarts, neighbors)
        else:
            named_walks = random_walks._anonymize(walks)
            text = random_walks._as_text(named_walks, restarts)
        encoded_input = model.model.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=model.model.max_length
        )
        encoded_input = encoded_input.to(target_ids.device)
        input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
        output = model.model.backbone(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        pred = model.model.head(output.last_hidden_state[:, 0, :]).argmax(-1)

    graph = batch[batch_idx]
    input_text = model.model.tokenizer.batch_decode(input_ids)
    tokenized_text = model.model.tokenizer.tokenize(input_text[batch_idx])
    if head_idx == 'all':
        attention = output.attentions[layer_idx][batch_idx].mean(0)
    else:
        attention = output.attentions[layer_idx][batch_idx][head_idx]
    norm = plt.Normalize(0, 1.4)

    colored_text = ""
    attention_scores = attention[0, 1:-1] / attention[0, 1:-1].max()
    for token, score in zip(tokenized_text[1:-1], attention_scores):
        if token == '#':
            token = '\\#'
        val = score.item()
        if val < 0.01:
            r, g, b = 1, 1, 1
        else:
            r, g, b = matplotlib.cm.Oranges(norm(val))[:3]
        latex_color = f"{int(r * 255)},{int(g * 255)},{int(b * 255)}"
        colored_text += (f"\\cctext{{{latex_color}}}{{{token}}}")
        if token in ('-', '\#', ';'):
            colored_text += "\\allowbreak"

    edge_scores = torch.zeros(graph.num_edges, dtype=attention.dtype, device=attention.device)
    edge_visited = torch.zeros(graph.num_edges, dtype=torch.bool, device=attention.device)
    edge_times = torch.zeros(graph.num_edges, dtype=torch.int32, device=pred.device)

    id2vertex = dict()
    vertex2id = dict()
    for id, vertex in zip(named_walks[0], walks[0]):
        if id in id2vertex:
            assert id2vertex[id] == vertex
        else:
            id2vertex[id] = vertex
        if vertex in vertex2id:
            assert vertex2id[vertex] == id
        else:
            vertex2id[vertex] = id

    edge2edgeid = dict()
    for i, (src, dst) in enumerate(graph.edge_index.unbind(1)):
        edge2edgeid[(src.item(), dst.item())] = i

    time = 0
    for i, query_token in enumerate(tokenized_text):
        if query_token == '[CLS]':
            assert i == 0
            prev = -1
            for j, key_token in enumerate(tokenized_text):
                if key_token in ('[CLS]', '[SEP]', '[PAD]'):
                    continue
                if tokenized_text[j + 1] in ('[SEP]', '[PAD]'):
                    break

                if key_token in ('-', '#', ';'):
                    cur = int(tokenized_text[j + 1])
                else:
                    cur = int(key_token)
                    if prev >= 0:
                        if not edge_visited[edge2edgeid[(id2vertex[prev], id2vertex[cur])]].item():
                            edge_visited[edge2edgeid[(id2vertex[prev], id2vertex[cur])]] = True
                            edge_visited[edge2edgeid[(id2vertex[cur], id2vertex[prev])]] = True
                            edge_times[edge2edgeid[(id2vertex[prev], id2vertex[cur])]] = time
                            edge_times[edge2edgeid[(id2vertex[cur], id2vertex[prev])]] = time
                        time += 1
                if prev >= 0:
                    edge_scores[edge2edgeid[(id2vertex[prev], id2vertex[cur])]] += attention[i, j]
                    edge_scores[edge2edgeid[(id2vertex[cur], id2vertex[prev])]] += attention[i, j]

                if key_token not in ('-', '#', ';') and (j == 1 or (tokenized_text[j - 1] == '-')):
                    prev = cur

    edge_color = (edge_scores / edge_scores.max()).cpu().detach().numpy()
    edge_times = (edge_times.float() / edge_times.float().max()).cpu().detach().numpy()

    G = nx.to_undirected(to_networkx(graph))
    pos = nx.circular_layout(G)
    if dataset_name == 'SR16':
        if graph.y[batch_idx].item() == 0:
            old2new = {0: 12, 1: 15, 2: 4, 3: 5, 4: 3, 5: 13, 6: 6, 7: 0, 8: 2, 9: 10, 10: 9, 11: 1, 12: 7, 13: 11, 14: 14, 15: 8}
        else:
            old2new = {0: 6, 1: 9, 2: 12, 3: 15, 4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 7, 10: 10, 11: 13, 12: 5, 13: 8, 14: 11, 15: 14}
        pos = {old2new[k]: v for k, v in pos.items()}
    for i, (src, dst, data) in enumerate(G.edges(data=True)):
        data['weight'] = edge_color[edge2edgeid[src, dst]]
        data['time'] = edge_times[edge2edgeid[src, dst]]

    if draw_graph:
        plt.figure(figsize=(6, 6))
        nx.draw_networkx(G, pos, node_size=180, node_color='w', edgecolors='k', font_size=8, font_color='k', width=0.8, with_labels=False)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}.pdf')

    plt.figure(figsize=(6, 6))
    nx.draw_networkx(G, pos, node_size=180, node_color='w', edgecolors='k', font_size=8, font_color='k', width=0.8, with_labels=False)
    nx.draw_networkx_labels(G, pos, font_size=8, labels=vertex2id, font_color='k')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_seed_{seed}.pdf')

    plt.figure(figsize=(6, 6))
    edges, weights = zip(*nx.get_edge_attributes(G, 'time').items())
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=3, edge_cmap=plt.cm.turbo)
    nx.draw_networkx_nodes(G, pos, node_size=180, node_color='w', edgecolors='k')
    nx.draw_networkx_labels(G, pos, font_size=8, labels=vertex2id, font_color='k')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_time_seed_{seed}.pdf')

    plt.figure(figsize=(6, 6))
    edge_list = list(G.edges(data=True))
    edge_list.sort(key=lambda x: x[2]['weight'])
    for edge in edge_list:
        val = edge[2]['weight']
        r, g, b = matplotlib.cm.Oranges(norm(val))[:3]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=3, edge_color=(r, g, b), alpha=(0 if val < 0.01 else 1))
    nx.draw_networkx_nodes(G, pos, node_size=180, node_color='w', edgecolors='k')
    nx.draw_networkx_labels(G, pos, font_size=8, labels=vertex2id, font_color='k')
    plt.axis('off')
    plt.tight_layout()
    if head_idx == 'all':
        plt.savefig(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_layer_{(layer_idx % num_layers) + 1}_head_all_seed_{seed}.pdf')
        with open(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_layer_{(layer_idx % num_layers) + 1}_head_all_seed_{seed}.txt', 'w') as f:
            f.write(colored_text)
    else:
        plt.savefig(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_layer_{(layer_idx % num_layers) + 1}_head_{(head_idx % num_heads) + 1}_seed_{seed}.pdf')
        with open(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_layer_{(layer_idx % num_layers) + 1}_head_{(head_idx % num_heads) + 1}_seed_{seed}.txt', 'w') as f:
            f.write(colored_text)

    plt.close('all')

    with open(Path(root) / f'graph_{graph.y[batch_idx].item() + 1}_pred_{pred[batch_idx].item() + 1}_walk_seed_{seed}.txt', 'w') as f:
        f.write(input_text[batch_idx])

# %%
root = f'../experiments/figures/{dataset_name}'
Path(root).mkdir(parents=True, exist_ok=True)
for seed in (0, 1, 2, 42):
    if dataset_name == 'SR16':
        data_range = range(2)
    elif dataset_name == 'SR25':
        data_range = range(15)
    elif dataset_name == 'CSL':
        data_range = reversed(range(0, 150, 15))
    for data_idx in data_range:
        for layer_idx in range(12):
            for head_idx in range(12):
                draw(root, data_idx, layer_idx, head_idx, seed, draw_graph=(layer_idx==0 and head_idx==0))
            draw(root, data_idx, layer_idx, 'all', seed, draw_graph=False)

# %%



