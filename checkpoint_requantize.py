# checkpoint_requantize.py (short version)
import sys, pickle, math
path = sys.argv[1] if len(sys.argv)>1 else "GA/ga_checkpoint.pkl"
K = int(sys.argv[2]) if len(sys.argv)>2 else 16
with open(path,"rb") as f: data = pickle.load(f)
pop = data["population"]
pop = [ind for ind in pop if getattr(ind,"fitness",None) and ind.fitness.valid]
pop.sort(key=lambda ind: float(ind.fitness.values[0]))
data["population"] = pop[:K]
ga = data.get("ga_state", {})
for k in ("mdf_data","alpha_data","results","labels",
          "all_gene_values_successful","all_gene_values_unsuccessful",
          "all_losses_successful","all_losses_unsuccessful"):
    if k in ga: ga[k] = []
ga["walker_history"] = {i: [] for i in range(len(data["population"]))}
with open(path,"wb") as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
print("Trimmed to", len(data["population"]))
