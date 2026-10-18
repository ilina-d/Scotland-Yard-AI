import torch
from torch import optim
from utils.neural_nets.detective_nets import BaseDetectiveNet
from utils.players import BasePlayer, TrainingDetectiveNetPlayer
from utils.game import Game
from torch.nn import functional as F
import os


class DetectiveTrainer:
    """ GNN Trainer for the Detective Players. """

    _training_logs: str = ''

    @classmethod
    def _log(cls, text: str) -> None:
        """ Write to the training logs. """

        cls._training_logs += f'{text}\n'
        print(text)


    @staticmethod
    def _play_episode(game: Game, update_freq: str) -> tuple[dict[str, list], str]:
        """ Play a full game or one round where all detectives are controlled by the same given net. """

        if update_freq == 'per_game':
            winner = game.play()

        elif update_freq == 'per_round':
            game._new_round_silently()
            winner = game.state.winner

        else:
            while True:
                game._new_round_silently()
                if game.state.winner or game.state.get_steps_until_reveal() == 1:
                    break
            winner = game.state.winner

        all_move_logs = {
            'conf': [],
            'value_before': [],
            'entropy': [],
            'acted' : []
        }

        while True:
            empty_count = 0
            for det in (game.detective_r, game.detective_g, game.detective_b, game.detective_o, game.detective_p):

                if not det.move_logs:
                    empty_count += 1
                    continue

                current_move = det.move_logs.pop(0)
                all_move_logs['conf'].append(current_move['conf'])
                all_move_logs['value_before'].append(current_move['value_before'])
                all_move_logs['entropy'].append(current_move['entropy'])
                all_move_logs['acted'].append(current_move['acted'])

            if empty_count == 5:
                break

        return all_move_logs, winner


    @classmethod
    def train(cls, detective_net: BaseDetectiveNet, lr: float, wd: float, epochs: int, opponent: BasePlayer,
              gamma: float, value_coef: float, entropy_coef: float, update_freq: str = 'per_game') -> None:
        """
        Train the Detective Net.

        Arguments:
            detective_net: The DetectiveNet to be trained.
            lr: Learning rate.
            wd: Weight decay.
            epochs: Number of epochs.
            opponent: The Mr. X player.
            gamma: Per-move discount factor.
            value_coef: Weight of the value head loss.
            entropy_coef: Weight of the entropy bonus (encourages exploration).
            update_freq: How often weight updates occur when training
                         (Options are: 'per_game', 'per_round' and 'per_reveal').
        """

        cls._training_logs = ''
        cls._log(f'{detective_net.__class__.__name__} Structure Details:\n{detective_net}\n\n')

        epoch_spaces = len(str(epochs))

        optimizer = optim.AdamW(detective_net.parameters(), lr = lr, weight_decay = wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)

        cls._log(
            f'{detective_net.__class__.__name__} Training Details:\n'
            f'* Opponent: {opponent.__class__.__name__}\n'
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

        cls._log(f'\nTraining with: {detective_net.device}...')
        detective_net.train()

        for epoch in range(1, epochs + 1):
            detective_r = TrainingDetectiveNetPlayer('r', detective_net)
            detective_g = TrainingDetectiveNetPlayer('g', detective_net)
            detective_b = TrainingDetectiveNetPlayer('b', detective_net)
            detective_o = TrainingDetectiveNetPlayer('o', detective_net)
            detective_p = TrainingDetectiveNetPlayer('p', detective_net)

            game = Game(
                detective_r = detective_r,
                detective_g = detective_g,
                detective_b = detective_b,
                detective_o = detective_o,
                detective_p = detective_p,
                mr_x = opponent,
                visuals = False,
                wait_after_move = None,
                always_show_x = False
            )

            if update_freq != 'per_game':
                game._step_x_silently()

            round_count = 0
            while True:
                move_logs, winner = cls._play_episode(game, update_freq)
                round_count += 1

                log_probs = torch.stack(move_logs['conf'])
                past_values = torch.stack(move_logs['value_before'])
                entropies = torch.stack(move_logs['entropy'])

                acted = torch.tensor(move_logs['acted'], dtype = torch.float32, device = past_values.device)
                num_acted = acted.sum().clamp(min = 1)

                if winner:
                    final_reward = 1.0 if winner == 'Detectives' else -1.0
                    winner_info = 'D' if winner == 'Detectives' else 'X'
                    next_values = torch.cat([past_values[1:].detach(), past_values.new_tensor([final_reward / gamma])])

                else:
                    winner_info = ' '
                    with torch.no_grad():
                        _, post_value = detective_net.get_processed_output(
                            game.state, player = 'r', value_as_tensor = True, value_only = True
                        )
                    next_values = torch.cat([past_values[1:].detach(), post_value.reshape(1)])

                targets = (gamma * next_values).detach()

                advantages = (targets - past_values).detach()
                policy_loss = -(log_probs * advantages * acted).sum() / num_acted
                value_loss = F.mse_loss(past_values, targets)
                entropy = (entropies * acted).sum() / num_acted
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

        detective_net.eval()
        cls._log('\nTraining finished.')


    @classmethod
    def save(cls, detective_net: BaseDetectiveNet, file_name: str, comment: str, logs: bool = True) -> None:
        """
        Save the weights of a trained Detective Net to a local file (at trained_nets/detective_nets/file_name.pt).

        Arguments:
            detective_net: The net whose weights are being saved.
            file_name: The name of the save file.
            comment: A comment to be saved along with the weights file.
            logs: Whether to include the training logs.
        """

        os.makedirs('trained_nets/detective_nets', exist_ok = True)
        torch.save(detective_net.state_dict(), f'trained_nets/detective_nets/{file_name}.pt')

        if logs:
            comment += f'\n\n{cls._training_logs}'

        with open(f'trained_nets/detective_nets/{file_name}_comment.txt', 'w') as file:
            file.write(comment)


__all__ = ['DetectiveTrainer']
