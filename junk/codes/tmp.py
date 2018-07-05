
























# Subset probability for plotting:
prob=prob_dict["dbn_30~105_120"]

# Get the order of the probabilities:
order_index = np.argsort(prob)

# Create temporal data frame for plotting:
tmp = pd.DataFrame({'Sorghum inbred lines': df.id_gbs[prob.index].iloc[order_index], 'Top 20% rank probabilities': prob.iloc[order_index].values}) 

# Plot probabilities barplot:
p1 = sns.barplot(x='Top 20% rank probabilities', y='Sorghum inbred lines', color='red', saturation=1, data=tmp)
p1.set(yticklabels=[])
plt.xlim(0, 1)
plt.show()








# Subset probability for plotting:
prob=prob_dict["dbn_30~105_120"]

# Get the order of the probabilities:
order_index = np.argsort(prob)[::-1]

# Subset probability for plotting:
p1 = prob.iloc[order_index].plot.barh(color='red')
p1.set(yticklabels=[])
plt.xlabel('Top 20% rank probabilities')
plt.ylabel('Sorghum inbred lines')
plt.xlim(0, 1)




















