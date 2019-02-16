import math
from lcztools import LeelaBoard, load_network
from collections import OrderedDict

from threading import Lock, Condition
from multiprocessing.dummy import Pool

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
        while cur is not top:
            moves.append(cur.move)
            cur = cur.parent
        return ' '.join(reversed(moves))

SEARCHER = None

def Threaded_UCT_search(board, num_reads, net=None, C=1.0):
    assert (net != None)
    global SEARCHER
    if not SEARCHER:
        SEARCHER = Searcher()

    return SEARCHER.do_search(board, num_reads, net, C)

VIRT_LOSS = -3
def add_loss(cur, loss):
    while cur.parent is not None:
        cur.number_visits -= loss
        cur.total_value += loss
        cur = cur.parent

LOCK = Lock()
COND = Condition(LOCK)

import logging
logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(levelname)s:%(funcName)s:%(lineno)d: %(message)s')
log = logging.getLogger('Searcher')

class Searcher():
    def __init__(self, threads=5, reuse_tree=False):
        self.pool = None
        self.threads = threads
        self.tree = None
        self.reuse_tree = reuse_tree

    def do_search(self, board, num_reads, net=None, C=1.0):
        if not self.pool:
            self.pool = Pool(self.threads)

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

        res = []
        for i in range(self.threads):
            res.append(self.pool.apply_async(self.search, args=(root, num_reads, net, C, i)))
        for r in res:
            print(r.get())

        self.tree = root

        return max(root.children.items(),
                    key=lambda item: (item[1].number_visits, item[1].Q()))

    def search(self, root, num_reads, net=None, C=1.0, searcher_num=0):
        leaf = None
        while 1:
            # get lock
            with LOCK as l:
                # from last search
                if leaf:
                    if leaf.parent:
                        leaf.parent.children[leaf.move] = leaf
                    add_loss(leaf, -VIRT_LOSS)
                    self.backup(leaf, leaf.total_value, leaf.number_visits)
                    leaf.locked = False

                if root.number_visits >= num_reads:
                    break

                leaf = self.select_leaf(root, C)
                if not leaf or leaf is root and searcher_num!=0:
                    log.debug('no leaf %s', searcher_num)
                    COND.wait()
                    continue

                log.debug('subtree search from %s', leaf.moves_from(root))
                add_loss(leaf, VIRT_LOSS)
                if leaf.parent:
                    leaf.parent.children.pop(leaf.move)
                COND.notify_all()
            self.search_subtree(leaf, 20, net, C)

        return max(root.children.items(),
                   key=lambda item: (item[1].number_visits, item[1].Q()))

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


#num_reads = 10000
#import time
#tick = time.time()
#UCT_search(GameState(), num_reads)
#tock = time.time()
#print("Took %s sec to run %s times" % (tock - tick, num_reads))
#import resource
#print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

import unittest
class TestSearch(unittest.TestCase):
    def test_search(self):
        net = load_network(backend='pytorch_cpu', filename='../weights_9149.txt.gz', policy_softmax_temp=2.2)
        b = LeelaBoard()
        #b = LeelaBoard(fen='3NQ3/7k/5K2/8/8/8/8/8 w - - 15 80')
        #b=LeelaBoard(fen='3N4/6Q1/5K2/7k/8/8/8/8 w - - 19 82')
        s = Searcher()
        root = UCTNode(b)
        ret = s.search(root, 2000, net, 3.4)
        print(ret)

if __name__ == '__main__':
    unittest.main()
