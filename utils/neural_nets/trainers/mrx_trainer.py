import torch
from torch import optim
from torch_geometric.utils import segment

from utils.neural_nets.mrx_nets import BaseMrXNet
from utils.players import BasePlayer, TrainingMrXNetPlayer
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

        elif update_freq == 'per_reveal':
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


    @classmethod
    def train(cls, mrx_net: BaseMrXNet, lr: float, wd: float, epochs: int, opponent_r: BasePlayer,
              opponent_g: BasePlayer, opponent_b: BasePlayer, opponent_o: BasePlayer, opponent_p: BasePlayer,
              gamma: float, value_coef: float, entropy_coef: float, update_freq: str = 'per_game') -> None:
        """
        Train the Mr. X Net.

        Arguments:
            mrx_net: The MrXNet to be trained.
            lr: Learning rate.
            wd: Weight decay.
            epochs: Number of epochs.
            opponent_r: The R Detective player.
            opponent_g: The G Detective player.
            opponent_b: The B Detective player.
            opponent_o: The O Detective player.
            opponent_p: The P Detective player.
            gamma: Per-move discount factor.
            value_coef: Weight of the value head loss.
            entropy_coef: Weight of the entropy bonus (encourages exploration).
            update_freq: How often weight updates occur when training
                         (Options are: 'per_game', 'per_round' and 'per_reveal').
        """

        cls._training_logs = ''
        cls._log(f'{mrx_net.__class__.__name__} Structure Details:\n{mrx_net}\n\n')

        epoch_spaces = len(str(epochs))

        optimizer = optim.AdamW(mrx_net.parameters(), lr = lr, weight_decay = wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)

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
            mr_x = TrainingMrXNetPlayer(mrx_net)

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

            round_count = 0
            while True:
                round_count += 1
                move_logs, winner = cls._play_episode(game, update_freq)

                log_probs = torch.stack(move_logs['conf'])
                past_values = torch.stack(move_logs['value_before'])
                entropies = torch.stack(move_logs['entropy'])

                if winner:
                    final_reward = -1.0 if winner == 'Detectives' else 1.0
                    winner_info = 'D' if winner == 'Detectives' else 'X'
                    next_values = torch.cat([past_values[1:].detach(), past_values.new_tensor([final_reward / gamma])])

                else:
                    winner_info = ' '
                    with torch.no_grad():
                        _, post_value = mrx_net.get_processed_output(
                            game.state, value_as_tensor = True, value_only = True
                        )
                    next_values = torch.cat([past_values[1:].detach(), post_value.reshape(1)])

                targets = (gamma * next_values).detach()

                advantages = (targets - past_values).detach()
                policy_loss = -(log_probs * advantages).mean()
                value_loss = F.mse_loss(past_values, targets)
                entropy = entropies.mean()
                loss = policy_loss + value_loss * value_coef - entropy * entropy_coef

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                segment_info = ''
                if update_freq == 'per_round':
                    segment_info = f'| Round: {round_count:<2} '
                elif update_freq == 'per_reveal':
                    segment_info = f'| Total steps: {game.state.x_step_count:<2} '

                cls._log(
                    f'* Epoch: {epoch:<{epoch_spaces}} '
                    f'{segment_info}'
                    f'| Winner: {winner_info} '
                    f'| Loss: {loss.item():<12.6f}'
                )

                if winner:
                    break

        mrx_net.eval()
        cls._log('\nTraining finished.')


    @classmethod
    def save(cls, mrx_net: BaseMrXNet, file_name: str, comment: str, logs: bool = True) -> None:
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

