import math
import queue

from chess import BLACK
from lcztools import LeelaBoard, load_network
from collections import OrderedDict

from multiprocessing import Pool, Process, Queue

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Searcher')

class UCTNode():
    def __init__(self, board=None, parent=None, prior=0):
        self.board = board
        self.is_expanded = False
        self.parents = [parent] if parent else []  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        if parent == None:
            self.total_value = 0.  # float
        else:
            self.total_value = -1.0
        self.number_visits = 0  # int
        self.terminal = False

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        return (math.sqrt(self.parents[0].number_visits)
                * self.prior / (1 + self.number_visits))

    def best_child(self, C):
        return max(self.children.items(),
                   key=lambda move_node: move_node[1].Q() + C*move_node[1].U())

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTNode(parent=self, prior=prior)

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
        if self.parents:
            move = next(m for m,c in self.parents[0].children.items() if c == self)
            return 'Move: {} Tot:{} Visits:{} prior:{} Q:{} U:{}'.format(move, self.total_value,
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
        while cur is not top and cur.parents:
            moves.append(cur.board.move_stack[-1].uci())
            cur = cur.parents[0]
        return ' '.join(reversed(moves))

def transposition_key(board):
    # add move to avoid hard cycles
    return board.pc_board._transposition_key() + (board.pc_board.fullmove_number,)

def check_cycle(node, seen=None):
    return False
    if not seen:
        seen = set()
    if node in seen:
        log.warning('found cycle %s', node.moves_from())
        import pdb
        pdb.set_trace()
        return True
    seen.add(node)
    for n in node.children.values():
        if check_cycle(n, seen.copy()):
            return True
    return False


def UCT_transpose_search(board, num_reads, net=None, C=1.0):
    log.info('got board: %s', repr(board))
    return Searcher().do_search(board, num_reads, net, C)

class Searcher():
    def __init__(self, reuse_tree=False, transpose_cache=None):
        self.tree = None
        self.reuse_tree = reuse_tree
        self.transpose_cache = transpose_cache if transpose_cache else {}

    def do_search(self, board, num_reads, net=None, C=1.0):
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
                        del root.parent
                        root.parent = None
            board.push(m)

        if not root:
            root = UCTNode(board.copy())

        root = self.search(root, num_reads + root.number_visits, net, C)

        self.tree = root

        return max(root.children.items(),
                    key=lambda item: (item[1].number_visits, item[1].Q()))

    def search(self, root, num_reads, net=None, C=1.0):
        for _ in range(num_reads):
            leaf, moves, is_transposition = self.select_leaf(root, C)
            if leaf.terminal:
                value_estimate = leaf.total_value
            elif not is_transposition:
                log.debug('got leaf %s %s', leaf, moves)
                board = leaf.board
                self.transpose_cache[transposition_key(board)] = leaf
                if board.pc_board.is_game_over() or board.is_draw():
                    # fix leelaboard - move stack is trimmed, so 3 fold doesnt work (override can_claim_threefold_repetition)
                    if board.is_draw():
                        value_estimate = 0.0
                    else:
                        result = board.pc_board.result(claim_draw=True)
                        value_estimate = {'1-0':1.0, '0-1':-1.0, '1/2-1/2': 0.0}[result]
                        if board.turn == BLACK:
                            value_estimate *= -1
                    leaf.terminal = True
                else:
                    child_priors, value_estimate = net.evaluate(leaf.board)
                    leaf.expand(child_priors)

            # in transpositions, only backup last parent, leaf already has value and was propogated to its other parents
            if is_transposition:
                self.backup(leaf.parents[-1], -value_estimate, 1)
            else:
                self.backup(leaf, value_estimate, 1)

        return root

    '''
    def expand(self, leaf, child_priors):
        leaf.is_expanded = True
        board = leaf.board.copy()
        for move, prior in child_priors.items():
            board.push_uci(move)
            existing = self.traspose_cache.get(board.pc_board._transposition_key())
            board.pop()
            if existing:
                log.debug('trasposition: %s  <=> %s', leaf.moves_from(), existing.moves_from())
                # FIXME: propagate counts up?
                leaf.children[move] = existing
                leaf.number_visits += existing.number_visits
                leaf.total_value += existing.total_value
            else:
                leaf.add_child(move, prior)
    '''

    def select_leaf(self, current, C):
        moves = []
        transposition = False
        move = None
        while current.is_expanded and current.children:
            move, current = current.best_child(C)
            moves.append(move)
            if len(moves) > 100:
                import pdb
                pdb.set_trace()

        if not current.board:
            current.board = current.parents[0].board.copy()
            current.board.push_uci(move)

            # check if this child is a transposition from another path
            key = transposition_key(current.board)
            existing = self.transpose_cache.get(key)
            if existing:
                log.info('found trasposition: %s  <=> %s', current.moves_from(), existing.moves_from())
                # transposition can be from any level
                # in that case makesure we're not building a cycle
                if current.board._lcz_transposition_counter[key] > 1:
                    log.info('got repetition')
                    current.total_value = 0.0
                    current.number_visits = 1
                    current.terminal = True
                else:
                    # link parent to existing, link existing to parent
                    current.parents[0].children[move] = existing
                    existing.parents.append(current.parents[0])
                    current = existing
                    transposition = True
                    check_cycle(current)

        return current, moves, transposition

    def backup(self, node: UCTNode, value_estimate: float, count: int=1):
        nodes = {node}
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        done = {}
        while nodes:
            parents = set()
            for cur in nodes:
                parents.update(cur.parents)  # FIXME: if we traspose across move numbers need to check done
                cur.number_visits += count
                cur.total_value += (value_estimate * turnfactor)
            nodes = parents
            turnfactor *= -1



if __name__ == '__main__':
    net = load_network(backend='pytorch_cpu', filename='../weights_9149.txt.gz', policy_softmax_temp=2.2)
    #b = LeelaBoard()
    b = LeelaBoard(fen='rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4')
    #b=LeelaBoard(fen='rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3')
    b=LeelaBoard(fen='8/8/8/5k2/3n1p2/3N3p/5K1P/8 b - - 1 54')
    s = Searcher()
    ret = s.do_search(b, 20000, net, 3.4)
    print(ret)
