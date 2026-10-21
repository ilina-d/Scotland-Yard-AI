import torch
from torch import optim
from utils.neural_nets.mrx_nets import BaseXNet
from utils.players import BasePlayer, TrainingXNetPlayer, RandomPlayer
from utils.game import Game
from torch.nn import functional as F
import os


class MrXTrainer:
    """ GNN Trainer for the Mr. X Players. """

    _training_logs: str = ''

    @classmethod
    def _log(cls, text: str) -> None:
        """ Write to the training logs. """

        cls._training_logs += f'{text}\n'
        print(text)


    @staticmethod
    def _play_episode(game: Game, update_freq: str) -> tuple[dict[str, list], str]:
        """ Play a full Scotland Yard game. """

        if update_freq == 'per_round':
            game._step_x_silently()
            game._step_d_silently()
            winner = game.state.winner

        elif update_freq == 'per_show':
            while True:
                game._step_x_silently()
                game._step_d_silently()
                if game.state.winner or game.state.get_steps_until_reveal() == 1:
                    break
            winner = game.state.winner
        else:
            winner = game.play()

        all_move_logs = {
            'conf' : [],
            'value_before' : [],
            'entropy' : []
        }

        while True:
            if not game.mr_x.move_logs:
                break

            current_move = game.mr_x.move_logs.pop(0)
            all_move_logs['conf'].append(current_move['conf'])
            all_move_logs['value_before'].append(current_move['value_before'])
            all_move_logs['entropy'].append(current_move['entropy'])

        return all_move_logs, winner


    @staticmethod
    def _compute_gae(values: torch.Tensor, final_reward: float, last_value: float,
                     gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        """ Helper function for computing Generalized Advantage Estimation. """

        move_count = len(values)
        advantages = torch.zeros(move_count)
        future_adv = 0.0

        for move in reversed(range(move_count)):
            if move == move_count - 1:
                new_val = last_value
                reward = final_reward
            else:
                new_val = values[move + 1]
                reward = 0.0

            surprise = reward + gamma * new_val - values[move]
            future_adv = surprise + gamma * lam * future_adv
            advantages[move] = future_adv

        targets = advantages + values
        return advantages, targets


    @classmethod
    def train(cls, mrx_net: BaseXNet, lr: float = 3e-4, wd: float = 1e-4, epochs: int = 10_000,
              opponent_r: BasePlayer = None, opponent_g: BasePlayer = None, opponent_b: BasePlayer = None,
              opponent_o: BasePlayer = None, opponent_p: BasePlayer = None, gamma: float = 0.99,
              value_coef: float = 0.5, entropy_coef: float = 0.003, update_freq: str = 'per_game',
              warmup_perc: float = 0.02, plateau_perc: float = 0.18) -> None:
        """
        Train the Mr. X Net.

        Arguments:
            mrx_net: The MrXNet to be trained.
            lr: Learning rate.
            wd: Weight decay.
            epochs: Number of epochs.
            opponent_r: The R Detective player (defaults to RandomPlayer).
            opponent_g: The G Detective player (defaults to RandomPlayer).
            opponent_b: The B Detective player (defaults to RandomPlayer).
            opponent_o: The O Detective player (defaults to RandomPlayer).
            opponent_p: The P Detective player (defaults to RandomPlayer).
            gamma: Per-move discount factor.
            value_coef: Weight of the value head loss.
            entropy_coef: Weight of the entropy bonus (encourages exploration).
            update_freq: How often weight updates occur when training
                         (Options are: 'per_game', 'per_round' and 'per_show').
            warmup_perc: The % of epochs that it takes to get to max lr.
            plateau_perc: The % of epochs that train at max lr.
        """

        epoch_spaces = len(str(epochs))

        if opponent_r is None:
            opponent_r = RandomPlayer(name = 'r')
        if opponent_g is None:
            opponent_g = RandomPlayer(name = 'g')
        if opponent_b is None:
            opponent_b = RandomPlayer(name = 'b')
        if opponent_o is None:
            opponent_o = RandomPlayer(name = 'o')
        if opponent_p is None:
            opponent_p = RandomPlayer(name = 'p')

        cls._training_logs = ''
        cls._log(f'{mrx_net.__class__.__name__} Structure Details:\n{mrx_net}\n\n')

        steps_per_episode = {'per_game': 1, 'per_show': 5, 'per_round': 24}[update_freq]
        total_steps = epochs * steps_per_episode
        warmup_end = int(warmup_perc * total_steps)
        plateau_end = int((warmup_perc + plateau_perc) * total_steps)
        final_frac = 0.1

        def lr_lambda(step):
            if step < warmup_end:
                return (step + 1) / warmup_end
            if step < plateau_end:
                return 1.0
            progress = min(1.0, (step - plateau_end) / max(1, total_steps - plateau_end))
            return 1.0 - (1.0 - final_frac) * progress

        optimizer = optim.AdamW(mrx_net.parameters(), lr = lr, weight_decay = wd)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        cls._log(
            f'{mrx_net.__class__.__name__} Training Details:\n'
            f'* Opponent R: {opponent_r.__class__.__name__}\n'
            f'* Opponent G: {opponent_g.__class__.__name__}\n'
            f'* Opponent B: {opponent_b.__class__.__name__}\n'
            f'* Opponent O: {opponent_o.__class__.__name__}\n'
            f'* Opponent P: {opponent_p.__class__.__name__}\n'
            f'* Learning Rate: {lr}\n'
            f'* Weight Decay: {wd}\n'
            f'* Epochs: {epochs}\n'
            f'\n'
            f'* Gamma: {gamma}\n'
            f'* Value Coef: {value_coef}\n'
            f'* Entropy Coef: {entropy_coef}\n'
            f'\n'
            f'* Update Frequency: {update_freq}\n'
        )

        cls._log(f'\nTraining with: {mrx_net.device}...')
        mrx_net.train()

        for epoch in range(1, epochs + 1):
            mr_x = TrainingXNetPlayer(mrx_net)

            game = Game(
                detective_r = opponent_r,
                detective_g = opponent_g,
                detective_b = opponent_b,
                detective_o = opponent_o,
                detective_p = opponent_p,
                mr_x = mr_x,
                visuals = False,
                wait_after_move = None,
                always_show_x = False
            )

            while True:
                move_logs, winner = cls._play_episode(game, update_freq)

                log_probs = torch.stack(move_logs['conf'])
                past_values = torch.stack(move_logs['value_before'])
                entropies = torch.stack(move_logs['entropy'])

                if winner:
                    final_reward = -1.0 if winner == 'Detectives' else 1.0
                    winner_info = 'D' if winner == 'Detectives' else 'X'
                    advantages, targets = cls._compute_gae(past_values.detach(), final_reward, 0.0, gamma, lam = 0.95)

                else:
                    final_reward = 0.0
                    winner_info = ''
                    with torch.no_grad():
                        _, post_value = mrx_net.get_processed_output(
                            game.state, value_as_tensor = True, value_only = True
                        )
                    advantages, targets = cls._compute_gae(
                        past_values.detach(), final_reward, post_value.item(), gamma, lam = 0.95
                    )

                policy_loss = -(log_probs * advantages).mean()
                value_loss = F.mse_loss(past_values, targets)
                entropy = entropies.mean()
                loss = policy_loss + value_loss * value_coef - entropy * entropy_coef

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                if winner:
                    break

            cls._log(
                f'* Epoch: {epoch:<{epoch_spaces}} '
                f'| Winner: {winner_info} '
                f'| LR: {scheduler.get_last_lr()[0]:<12.6f} '
                f'| Loss: {loss.item():<12.6f}'
            )

        mrx_net.eval()
        cls._log('\nTraining finished.')


    @classmethod
    def save(cls, mrx_net: BaseXNet, file_name: str, comment: str, logs: bool = True) -> None:
        """
        Save the weights of a trained Mr. X Net to a local file (at trained_nets/mrx_nets/file_name.pt).

        Arguments:
            mrx_net: The net whose weights are being saved.
            file_name: The name of the save file.
            comment: A comment to be saved along with the weights file.
            logs: Whether to include the training logs.
        """

        os.makedirs('trained_nets/mrx_nets', exist_ok=True)
        torch.save(mrx_net.state_dict(), f'trained_nets/mrx_nets/{file_name}.pt')

        if logs:
            comment += f'\n\n{cls._training_logs}'

        with open(f'trained_nets/mrx_nets/{file_name}_comment.txt', 'w') as file:
            file.write(comment)


__all__ = ['MrXTrainer']
