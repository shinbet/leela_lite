import math
import queue

from lcztools import LeelaBoard, load_network
from collections import OrderedDict

from multiprocessing import Pool, Process, Queue

import logging
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(levelname)s:%(funcName)s:%(lineno)d: %(message)s')
log = logging.getLogger('Searcher')

class UCTNode():
    def __init__(self, board=None, parent=None, move=None, prior=0):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        if parent == None:
            self.total_value = 0.  # float
        else:
            self.total_value = -1.0
        self.number_visits = 0  # int
        self.locked = False

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        return (math.sqrt(self.parent.number_visits)
                * self.prior / (1 + self.number_visits))

    def best_child(self, C):
        return max(self.children.values(),
                   key=lambda node: node.Q() + C*node.U())

    def select_leaf(self, C):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child(C)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def backup(self, value_estimate: float, count: int=1):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:
            current.number_visits += count
            current.total_value += (value_estimate * turnfactor)
            current = current.parent
            turnfactor *= -1
        current.number_visits += count
        current.total_value += (value_estimate * turnfactor)

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTNode(parent=self, move=move, prior=prior)

    def dump(self, move, C):
        print("---")
        print("move: ", move)
        print("total value: ", self.total_value)
        print("visits: ", self.number_visits)
        print("prior: ", self.prior)
        print("Q: ", self.Q())
        print("U: ", self.U())
        print("BestMove: ", self.Q() + C * self.U())
        #print("math.sqrt({}) * {} / (1 + {}))".format(self.parent.number_visits,
        #      self.prior, self.number_visits))
        print("---")

    def __str__(self):
        if self.parent:
            return 'Move: {} Tot:{} Visits:{} prior:{} Q:{} U:{}'.format(self.move, self.total_value,
                                                                         self.number_visits, self.prior,
                                                                         self.Q(), self.U())
        else:
            return 'root node Tot:{} Visits:{}'.format(self.total_value, self.number_visits)

    def dump_children(self):
        for move, c in self.children.values():
            print(c)

    def moves_from(self, top=None):
        cur = self
        moves = []
        while cur is not top and cur.move:
            moves.append(cur.move)
            cur = cur.parent
        return ' '.join(reversed(moves))

SEARCHER = None

def mp_search(board, num_reads, net=None, C=1.0):
    assert (net != None)
    global SEARCHER
    if not SEARCHER:
        SEARCHER = Searcher()

    return SEARCHER.do_search(board, num_reads, net, C)

def add_loss(cur, loss):
    while cur is not None:
        cur.number_visits -= loss
        cur.total_value += loss
        cur = cur.parent

def worker(input, output, nn):
    while True:
        board, num_reads, C = input.get()
        root = UCTNode(board)
        search_subtree(root, num_reads, nn, C)
        output.put(root)


def search_subtree(root, num_reads, net=None, C=1.0):
    for _ in range(num_reads):
        leaf = root.select_leaf(C)
        #log.info('got leaf %s', leaf)
        board = leaf.board
        if board.pc_board.is_game_over() or board.is_draw():
            result = board.pc_board.result(claim_draw=True)
            value_estimate = {'1-0':1.0, '0-1':-1.0, '1/2-1/2': 0.0}[result]
        else:
            child_priors, value_estimate = net.evaluate(leaf.board)
            leaf.expand(child_priors)
            #leaf.total_value = 0.0
        leaf.backup(value_estimate, 1)

    if root.children:
        return max(root.children.items(),
                   key=lambda item: (item[1].number_visits, item[1].Q()))


class Searcher():
    def __init__(self, threads=4, reuse_tree=True):
        self.pool = []
        self.threads = threads
        self.tree = None
        self.reuse_tree = reuse_tree
        self.virt_loss = -3

    def do_search(self, board, num_reads, net=None, C=1.0):
        if not self.pool:
            self.task_queue = Queue()
            self.done_queue = Queue()
            for _ in range(self.threads):
                self.pool.append(Process(target=worker, args=(self.task_queue, self.done_queue, net)).start())

        root = None
        if self.reuse_tree and self.tree and board.move_stack:
            # see if we have it in our cache

            # LeelaBoard trims move history :(
            # just check one move up
            m = board.pop()
            if board == self.tree.board:
                root = self.tree.children.get(m.uci())
                if root:
                    log.debug('found tree with %s nodes', root.number_visits)
                    # trim tree
                    if root.parent:
                        del root.parent.children[root.move]
                        root.parent = None
                root.move = None
            board.push(m)
            '''
            new_moves = board.move_stack
            our_moves = self.tree.board.move_stack
            if len(our_moves) <= len(new_moves) and our_moves == new_moves[:len(our_moves)]:
                root = self.tree
                for m in new_moves[len(our_moves):]:
                    if root.children and m.uci() in root.children:
                        root = root.children[m.uci()]
                    else:
                        root = None
                        break
                else:
                    log.debug('found tree with %s nodes', root.number_visits)
                    # trim tree
                    if root.parent:
                        del root.parent.children[root.move]
                        root.parent = None
        '''

        if not root:
            root = UCTNode(board.copy())

        root = self.search(root, num_reads + root.number_visits, C)

        print(root)
        self.tree = root

        return max(root.children.items(),
                    key=lambda item: (item[1].number_visits, item[1].Q()))

    def search(self, root, num_reads, C=1.0):

        pending = 0
        need = len(self.pool)
        leaf_to_parent = {}

        while root.number_visits < num_reads:
            while pending < need:
                leaf = self.select_leaf(root, C)
                if leaf:
                    # special case of first node
                    if leaf == root and pending == 1:
                        break

                    log.debug('subtree search from %s', leaf.moves_from())
                    self.task_queue.put((leaf.board, 50, C))
                    leaf.locked = True
                    leaf_to_parent[leaf.board] = leaf
                    if leaf.parent:
                        del leaf.parent.children[leaf.move]
                        if self.virt_loss:
                            add_loss(leaf.parent, self.virt_loss)
                    pending += 1
                else:
                    break

            while pending:
                leaf = self.done_queue.get()
                orig_leaf = leaf_to_parent.pop(leaf.board)
                if orig_leaf.parent:
                    # reattach to tree
                    orig_leaf.parent.children[orig_leaf.move] = leaf
                    leaf.parent = orig_leaf.parent
                    leaf.move = orig_leaf.move
                    leaf.prior = orig_leaf.prior

                    if self.virt_loss:
                        # undo loss
                        add_loss(leaf.parent, -self.virt_loss)
                    # backup values
                    self.backup(leaf, leaf.total_value, leaf.number_visits)
                else:
                    # first node
                    root = leaf

                pending -= 1
                log.debug('subtree done %s', leaf.moves_from())

        return root

    def select_leaf(self, current, C):
        while current.is_expanded and current.children:
            current = current.best_child(C)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def backup(self, node: UCTNode, value_estimate: float, count: int):
        current = node.parent
        turnfactor = -1
        while current is not None:
            current.number_visits += count
            current.total_value += (value_estimate *
                                    turnfactor)
            current = current.parent
            turnfactor *= -1

    def search_subtree(self, root, num_reads, net=None, C=1.0):
        for _ in range(num_reads):
            leaf = self.select_leaf(root, C)
            #log.info('got leaf %s', leaf)
            board = leaf.board
            if board.pc_board.is_game_over() or board.is_draw():
                result = board.pc_board.result(claim_draw=True)
                value_estimate = {'1-0':1.0, '0-1':-1.0, '1/2-1/2': 0.0}[result]
            else:
                child_priors, value_estimate = net.evaluate(leaf.board)
                leaf.expand(child_priors)
            self.backup_subtree(root, leaf, value_estimate)

        if root.children:
            return max(root.children.items(),
                       key=lambda item: (item[1].number_visits, item[1].Q()))

    def backup_subtree(self, root: UCTNode, leaf: UCTNode, value_estimate: float):
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        current = leaf
        while current is not root:
            current.number_visits += 1
            current.total_value += (value_estimate *
                                    turnfactor)
            current = current.parent
            turnfactor *= -1
        current.number_visits += 1
        current.total_value += (value_estimate *
                                turnfactor)

import unittest
class TestSearch(unittest.TestCase):
    def test_search(self):
        net = load_network(backend='pytorch_cpu', filename='../weights_9149.txt.gz', policy_softmax_temp=2.2)
        b = LeelaBoard()
        #b = LeelaBoard(fen='3NQ3/7k/5K2/8/8/8/8/8 w - - 15 80')
        #b=LeelaBoard(fen='3N4/6Q1/5K2/7k/8/8/8/8 w - - 19 82')
        s = Searcher()
        ret = s.do_search(b, 2000, net, 3.4)
        print(ret)

if __name__ == '__main__':
    unittest.main()
