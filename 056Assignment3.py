from collections import defaultdict, deque
import heapq

class BuildingRecord:
    def __init__(self, building_id, name, location):
        self.building_id = building_id
        self.name = name
        self.location = location

class BSTNode:
    def __init__(self, record):
        self.record = record
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
    def insertBuilding(self, record):
        self.root = self._insert(self.root, record)
    def _insert(self, node, record):
        if node is None: return BSTNode(record)
        if record.building_id < node.record.building_id:
            node.left = self._insert(node.left, record)
        else:
            node.right = self._insert(node.right, record)
        return node
    def searchBuilding(self, building_id):
        node = self._search(self.root, building_id)
        return node.record if node else None
    def _search(self, node, building_id):
        if node is None: return None
        if node.record.building_id == building_id: return node
        if building_id < node.record.building_id: return self._search(node.left, building_id)
        return self._search(node.right, building_id)
    def height(self): return self._height(self.root)
    def _height(self, node): return 0 if not node else 1+max(self._height(node.left),self._height(node.right))
    def inorder(self): r=[]; self._inorder(self.root,r); return r
    def _inorder(self,node,r):
        if node: self._inorder(node.left,r); r.append(node.record); self._inorder(node.right,r)
    def preorder(self): r=[]; self._preorder(self.root,r); return r
    def _preorder(self,node,r):
        if node: r.append(node.record); self._preorder(node.left,r); self._preorder(node.right,r)
    def postorder(self): r=[]; self._postorder(self.root,r); return r
    def _postorder(self,node,r):
        if node: self._postorder(node.left,r); self._postorder(node.right,r); r.append(node.record)

class AVLNode:
    def __init__(self, record):
        self.record = record
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self): self.root=None
    def insertBuilding(self, record): self.root=self._insert(self.root,record)
    def _height(self,node): return node.height if node else 0
    def height(self): return self._height(self.root)
    def _balance(self,node): return 0 if not node else self._height(node.left)-self._height(node.right)
    def _rightRotate(self,y):
        x=y.left;T2=x.right;x.right=y;y.left=T2
        y.height=1+max(self._height(y.left),self._height(y.right))
        x.height=1+max(self._height(x.left),self._height(x.right))
        return x
    def _leftRotate(self,x):
        y=x.right;T2=y.left;y.left=x;x.right=T2
        x.height=1+max(self._height(x.left),self._height(x.right))
        y.height=1+max(self._height(y.left),self._height(y.right))
        return y
    def _insert(self,node,record):
        if not node: return AVLNode(record)
        if record.building_id<node.record.building_id: node.left=self._insert(node.left,record)
        else: node.right=self._insert(node.right,record)
        node.height=1+max(self._height(node.left),self._height(node.right))
        balance=self._balance(node)
        if balance>1 and record.building_id<node.left.record.building_id:
            print("Rotation: LL at", node.record.building_id)
            return self._rightRotate(node)
        if balance<-1 and record.building_id>node.right.record.building_id:
            print("Rotation: RR at", node.record.building_id)
            return self._leftRotate(node)
        if balance>1 and record.building_id>node.left.record.building_id:
            print("Rotation: LR at", node.record.building_id)
            node.left=self._leftRotate(node.left); return self._rightRotate(node)
        if balance<-1 and record.building_id<node.right.record.building_id:
            print("Rotation: RL at", node.record.building_id)
            node.right=self._rightRotate(node.right); return self._leftRotate(node)
        return node
    def searchBuilding(self,building_id): n=self._search(self.root,building_id); return n.record if n else None
    def _search(self,node,building_id):
        if not node: return None
        if node.record.building_id==building_id: return node
        if building_id<node.record.building_id: return self._search(node.left,building_id)
        return self._search(node.right,building_id)
    def inorder(self): r=[]; self._inorder(self.root,r); return r
    def _inorder(self,node,r):
        if node: self._inorder(node.left,r); r.append(node.record); self._inorder(node.right,r)
    def preorder(self): r=[]; self._preorder(self.root,r); return r
    def _preorder(self,node,r):
        if node: r.append(node.record); self._preorder(node.left,r); self._preorder(node.right,r)
    def postorder(self): r=[]; self._postorder(self.root,r); return r
    def _postorder(self,node,r):
        if node: self._postorder(node.left,r); self._postorder(node.right,r); r.append(node.record)

class CampusGraph:
    def __init__(self,vertex_count):
        self.V=vertex_count
        self.adj_list=defaultdict(list)
        self.adj_matrix=[[0]*vertex_count for _ in range(vertex_count)]
        self.edges=[]
    def addEdge(self,u,v,w=1,undirected=True):
        self.adj_list[u].append((v,w))
        if undirected: self.adj_list[v].append((u,w))
        self.adj_matrix[u][v]=w
        if undirected: self.adj_matrix[v][u]=w
        self.edges.append((u,v,w))
        if undirected: self.edges.append((v,u,w))
    def bfs(self,start):
        visited=[False]*self.V;order=[];q=deque([start]);visited[start]=True
        while q:
            u=q.popleft();order.append(u)
            for v,_ in self.adj_list[u]:
                if not visited[v]: visited[v]=True;q.append(v)
        return order
    def dfs(self,start):
        visited=[False]*self.V;order=[]
        def _dfs(u):
            visited[u]=True;order.append(u)
            for v,_ in self.adj_list[u]:
                if not visited[v]: _dfs(v)
        _dfs(start);return order

def dijkstra(graph,src):
    dist=[float("inf")]*graph.V;dist[src]=0;pq=[(0,src)]
    while pq:
        d_u,u=heapq.heappop(pq)
        if d_u>dist[u]: continue
        for v,w in graph.adj_list[u]:
            if dist[v]>dist[u]+w:
                dist[v]=dist[u]+w;heapq.heappush(pq,(dist[v],v))
    return dist

class DisjointSet:
    def __init__(self,n): self.parent=list(range(n));self.rank=[0]*n
    def find(self,u):
        if self.parent[u]!=u: self.parent[u]=self.find(self.parent[u])
        return self.parent[u]
    def union(self,u,v):
        ru,rv=self.find(u),self.find(v)
        if ru==rv: return False
        if self.rank[ru]<self.rank[rv]: self.parent[ru]=rv
        elif self.rank[ru]>self.rank[rv]: self.parent[rv]=ru
        else: self.parent[rv]=ru;self.rank[ru]+=1
        return True

def kruskal_MST(vertex_count,edges):
    unique=[];seen=set()
    for u,v,w in edges:
        key=tuple(sorted((u,v)))+(w,)
        if key not in seen: unique.append((u,v,w));seen.add(key)
    ds=DisjointSet(vertex_count);mst=[];total=0
    unique.sort(key=lambda x:x[2])
    for u,v,w in unique:
        if ds.union(u,v): mst.append((u,v,w));total+=w
    return mst,total

class ExprNode:
    def __init__(self,value): self.value=value;self.left=None;self.right=None

def evaluateExpression(root):
    if root.left is None and root.right is None: return float(root.value)
    lv=evaluateExpression(root.left);rv=evaluateExpression(root.right);op=root.value
    if op=='+': return lv+rv
    if op=='-': return lv-rv
    if op=='*': return lv*rv
    if op=='/':
        if rv==0: raise ZeroDivisionError("division by zero")
        return lv/rv

class CampusPlannerSystem:
    def __init__(self,use_avl=True):
        self.buildings={}
        self.building_tree=AVLTree() if use_avl else BST()
        self.campus_graph=None
        self.expression_root=None
        self.use_avl=use_avl
    def addBuildingRecord(self,building_id,name,location):
        rec=BuildingRecord(building_id,name,location)
        self.buildings[building_id]=rec
        self.building_tree.insertBuilding(rec)
        return rec
    def listCampusLocations(self):
        return {"inorder":self.building_tree.inorder(),"preorder":self.building_tree.preorder(),"postorder":self.building_tree.postorder()}
    def constructCampusGraph(self,vertex_count,edges):
        self.campus_graph=CampusGraph(vertex_count)
        for u,v,w in edges: self.campus_graph.addEdge(u,v,w,undirected=True)
        return self.campus_graph
    def findOptimalPath(self,src): return dijkstra(self.campus_graph,src)
    def planUtilityLayout(self):
        mst,total=kruskal_MST(self.campus_graph.V,self.campus_graph.edges)
        return {"mst_edges":mst,"total_weight":total}
    def setExpressionTree(self,root): self.expression_root=root
    def evaluateEnergyExpression(self): return evaluateExpression(self.expression_root)

def print_records(label,records):
    print(f"\n{label}:")
    for r in records: print(f"ID={r.building_id}, Name={r.name}, Location={r.location}")

def print_matrix(matrix):
    print("\nAdjacency Matrix:")
    for row in matrix: print(" ".join(f"{w:3d}" for w in row))

def print_edges(edges):
    return ", ".join([f"({u}-{v}, w={w})" for (u, v, w) in edges])

def build_expression_tree_from_input():
    print("\nEnter expression for energy bill as postfix (tokens space-separated).")
    print("Allowed operators: + - * / ; numbers only as operands.")
    tokens=input("Postfix expression: ").strip().split()
    stack=[]
    for t in tokens:
        node=ExprNode(t)
        if t in ['+','-','*','/']:
            if len(stack)<2:
                raise ValueError("invalid postfix expression")
            node.right=stack.pop()
            node.left=stack.pop()
            stack.append(node)
        else:
            float(t)
            stack.append(node)
    if len(stack)!=1: raise ValueError("invalid postfix expression")
    return stack[0]

def main():
    print("=== Campus Navigation and Utility Planner (User Input) ===")
    n=int(input("Enter number of buildings: "))
    print("Choose tree type: 1) BST  2) AVL")
    choice=int(input("Your choice (1/2): "))
    use_avl = (choice==2)
    bst=BST(); avl=AVLTree()
    system=CampusPlannerSystem(use_avl=use_avl)
    print("\nEnter building records:")
    ids=[]
    for i in range(n):
        bid=int(input(f"Building {i+1} ID: "))
        name=input("Name: ")
        loc=input("Location: ")
        rec=BuildingRecord(bid,name,loc)
        bst.insertBuilding(rec)
        avl.insertBuilding(BuildingRecord(bid,name,loc))
        system.addBuildingRecord(bid,name,loc)
        ids.append(bid)

    print(f"\nBST Height: {bst.height()}")
    print(f"AVL Height: {avl.height()}")

    print_records("BST Inorder", bst.inorder())
    print_records("BST Preorder", bst.preorder())
    print_records("BST Postorder", bst.postorder())

    print_records("AVL Inorder", avl.inorder())
    print_records("AVL Preorder", avl.preorder())
    print_records("AVL Postorder", avl.postorder())

    print("\nEnter graph edges (undirected). Buildings indexed 0..n-1 in insertion order.")
    idx_map={i: ids[i] for i in range(n)}
    edges=[]
    m=int(input("Enter number of paths: "))
    for k in range(m):
        u=int(input("From index (0..n-1): "))
        v=int(input("To index (0..n-1): "))
        w=int(input("Distance (weight): "))
        edges.append((u,v,w))
    campus_graph=system.constructCampusGraph(vertex_count=n, edges=edges)
    print_matrix(campus_graph.adj_matrix)

    start=int(input("\nStart index for BFS/DFS/Dijkstra: "))
    bfs_order=campus_graph.bfs(start); dfs_order=campus_graph.dfs(start)
    print("BFS:", bfs_order)
    print("DFS:", dfs_order)

    dist=system.findOptimalPath(src=start)
    print("\nShortest distances from start:")
    for i,d in enumerate(dist):
        print(f"To index {i} (ID {idx_map[i]}): {d}")

    mst=system.planUtilityLayout()
    print("\nMST edges (Kruskal):")
    print(print_edges(mst["mst_edges"]))
    print(f"Total cable length: {mst['total_weight']}")

    print("\nExpression Tree for Energy Bill:")
    try:
        expr_root=build_expression_tree_from_input()
        system.setExpressionTree(expr_root)
        val=system.evaluateEnergyExpression()
        print("Energy bill result:", val)
    except Exception as e:
        print("Expression error:", e)

    q=int(input("\nSearch test: enter a Building ID to search: "))
    print("BST Search:", bst.searchBuilding(q))
    print("AVL Search:", avl.searchBuilding(q))
    if use_avl:
        print("System Tree (AVL) Search:", system.building_tree.searchBuilding(q))
    else:
        print("System Tree (BST) Search:", system.building_tree.searchBuilding(q))

if __name__=="__main__":
    main()
