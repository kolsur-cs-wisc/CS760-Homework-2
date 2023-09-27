import matplotlib.pyplot as plt
import graphviz

x1 = [0.2, 0.6, 0.2, 0.6]
x2 = [0.1, 0.9, 0.9, 0.1]

plt.scatter(x1[0:2], x2[0:2], label="y = 0")
plt.scatter(x1[2:4], x2[2:4], label="y = 1")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend()
plt.show()

tree = graphviz.Digraph()
tree.node("x1 >= 0.6 (1)")
tree.node("x2 >= 0.9 (2)")
tree.node("x2 >= 0.9 (5)")
tree.node("Y = 0 (3)")
tree.node("Y = 1 (4)")
tree.node("Y = 0 (6)")
tree.node("Y = 1 (7)")

tree.edge("x1 >= 0.6 (1)", "x2 >= 0.9 (2)")
tree.edge("x1 >= 0.6 (1)", "x2 >= 0.9 (5)")
tree.edge("x2 >= 0.9 (2)", "Y = 0 (3)")
tree.edge("x2 >= 0.9 (2)", "Y = 1 (4)")
tree.edge("x2 >= 0.9 (5)", "Y = 0 (6)")
tree.edge("x2 >= 0.9 (5)", "Y = 1 (7)")

tree.render(f'forced_split_tree', view=True, format='png')