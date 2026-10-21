from multiprocessing import cpu_count, get_context
import cProfile
import time
from utils.game import Game
from utils.players import BasePlayer, UserPlayer


def _get_game_data(args: dict[str, ...]) -> dict[str, ...]:
    """ Helper function for extracting data from a game simulation. """

    det_r, det_g, det_b, det_o, det_p = args['det_r'], args['det_g'], args['det_b'], args['det_o'], args['det_p']
    mr_x = args['mr_x']

    game = Game(
        detective_r = det_r,
        detective_g = det_g,
        detective_b = det_b,
        detective_o = det_o,
        detective_p = det_p,
        mr_x = mr_x,
        visuals = False,
        wait_after_move = None,
        always_show_x = False
    )

    game_start_time = time.time()
    winner = game.play()
    game_time_taken = time.time() - game_start_time

    return {
        'winner' : 'D' if winner == 'Detectives' else 'X',
        'time_taken' : game_time_taken,
        'step_count' : game.state.x_step_count
    }


class Simulator:
    """ Game simulator for win-rate comparisons. """

    def __init__(self, detective_r: BasePlayer, detective_g: BasePlayer, detective_b: BasePlayer,
                 detective_o: BasePlayer, detective_p: BasePlayer, mr_x: BasePlayer, num_sims: int,
                 measure_performance: bool = True, num_workers: int = 1) -> None:
        """
        Create a new Simulator instance.

        Arguments:
            detective_r: The player that controls the red detective.
            detective_g: The player that controls the green detective.
            detective_b: The player that controls the blue detective.
            detective_o: The player that controls the orange detective.
            detective_p: The player that controls the purple detective.
            mr_x: The player that controls Mr. X.
            num_sims: Number of games to simulate.
            measure_performance: Whether to measure code performance.
            num_workers: Number of cores to use while running simulations.
        """

        if any([isinstance(player, UserPlayer)
                for player in (detective_r, detective_g, detective_b, detective_o, detective_p, mr_x)]):
            raise Exception('Players cannot be of type UserPlayer when running simulations.')

        self.detective_r = detective_r
        self.detective_g = detective_g
        self.detective_b = detective_b
        self.detective_o = detective_o
        self.detective_p = detective_p
        self.mr_x = mr_x
        self.num_sims = num_sims
        self.measure_performance = measure_performance
        self.num_workers = min(cpu_count(), num_workers)


    def _run_sims(self) -> str:
        """ Helper function for running simulations. """

        _sim_args = [{
            'det_r' : self.detective_r, 'det_g' : self.detective_g, 'det_b' : self.detective_b,
            'det_o' : self.detective_o, 'det_p' : self.detective_p, 'mr_x' : self.mr_x
        } for _ in range(self.num_sims)]

        total_sim_time = time.time()
        sum_game_times = 0
        sum_game_turns = 0
        d_wins, x_wins = 0, 0

        print('[ SIMULATOR ] : Running simulations...', end = '\r')
        context = get_context('spawn')
        with context.Pool(processes = self.num_workers) as pool:
            try:
                for n, result in enumerate(pool.imap_unordered(_get_game_data, _sim_args), start = 1):
                    winner = result['winner']
                    sum_game_times += result['time_taken']
                    sum_game_turns += result['step_count']

                    if winner == 'X':
                        x_wins += 1
                    else:
                        d_wins += 1

                    percent_done = (n / self.num_sims) * 100
                    estimated_tl = (sum_game_times / n) * (self.num_sims - n)
                    estimated_tl = estimated_tl / self.num_workers
                    if estimated_tl == 0:
                        estimated_tl = 'done'
                    elif estimated_tl > 3600:
                        estimated_tl = f'~{estimated_tl // 3600} hours'
                    elif estimated_tl > 60:
                        estimated_tl = f'~{estimated_tl // 60} minutes'
                    else:
                        estimated_tl = f'~{round(estimated_tl)} seconds'

                    print(
                        f'\r\033[K'
                        f'[ SIMULATOR ] : Running simulations... '
                        f'{str(round(percent_done, 2)) + "%":<6} '
                        f'<{"=" * int(percent_done)}{"-" * int(100 - int(percent_done))}> '
                        f'| X({x_wins}) <-> ({d_wins})D '
                        f'| ETL: {estimated_tl}', end = ''
                    )
            except Exception:
                pool.terminate()
            finally:
                pool.close()
                pool.join()

        results = (
            f'\n'
            f'------[ SIMULATION RESULTS ]------\n'
            f'==================================\n'
            f'|> Players\n'
            f'* Detective Red    : {self.detective_r.__class__.__name__}\n'
            f'* Detective Green  : {self.detective_g.__class__.__name__}\n'
            f'* Detective Blue   : {self.detective_b.__class__.__name__}\n'
            f'* Detective Orange : {self.detective_o.__class__.__name__}\n'
            f'* Detective Purple : {self.detective_p.__class__.__name__}\n'
            f'* Mr. X            : {self.mr_x.__class__.__name__}\n'
            f'==================================\n'
            f'|> Duration\n'
            f'* Total Sim Time   : {time.time() - total_sim_time:.2f} sec\n'
            f'* Total Game Time  : {sum_game_times:.2f} sec\n'
            f'* Avg Game Time    : {sum_game_times / self.num_sims:.2f} sec\n'
            f'* Avg Game Steps   : {sum_game_turns / self.num_sims:.2f} x steps\n'
            f'==================================\n'
            f'|> Win Rates\n'
            f'* Detectives       : {d_wins} ( {d_wins / self.num_sims * 100:.2f}% )\n'
            f'* Mr. X            : {x_wins} ( {x_wins / self.num_sims * 100:.2f}% )\n'
            f'==================================\n'
        )

        print(results)
        return results


    def start(self) -> str:
        """ Run the simulator. """

        if self.measure_performance:
            profile = cProfile.Profile()
            results = profile.runcall(self._run_sims)
            profile.print_stats()

        else:
            results = self._run_sims()

        return results


__all__ = ['Simulator']
