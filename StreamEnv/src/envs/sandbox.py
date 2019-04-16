import numpy as np



class SandboxEnv():
    """
    
    Given a portfolio distribution:
    [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1]

    Calculate the commission of transitioning to:
    [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]

    Given the future price vector
    [1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]

    Delta Vector (target_vector-current_vector):
    [0.0, 0.0, 0.0, -0.1, 0.1, 0.0, 0.0, 0.0]

    where the permutations of change could be:
        1) SELL BASE1QUOTE 0.1 * current price
        2) BUY BASE2QUOTE 0.1 * current price

    or provided exists:
        1) SELL BASE1BASE2 0.1 * current price
    
    """

    def __init__(
        self,
    ):
        pass

    # Logic
    # ==========================================================>

    async def step(self):
        
        # future_price
        # future omega = (future price * action)/sum(future_price*action)
        # pv_vector = sum(future_price * action) * 
        #               ( 
        #                   1 - sum(
        #                           abs(
        #                               action - future_omega
        #                           )
        #                       ) * commission
        #               )    

        pass

    # Portfolio
    # ==========================================================>

    async def _act(self, action, future_price, commission):
        """
        
        """
        future_value = future_price * action
        total_future_value = np.sum(future_value)
        future_omega =  future_value / total_future_value
        pv_vector = total_future_value * (1-np.sum(np.absolute(action-future_omega)*commission))

        portfolio_value = np.prod(pv_vector)
        portfolio_mean = np.mean(pv_vector)
        portfolio_log_mean = np.mean(np.log(pv_vector))
        portfolio_std_dev = np.std(pv_vector)


    async def _derive_tnorm(self):
        """ 
        derive tnorm determines the total value of the current porfolio
        with respect to the quote asset at a given step.
        """
        pass

    async def _derive_pv(self):
        """ 
        calculates a vector representative of the distribution of value across
        a portfolio with respect to the quote asset at a given step.
        """
        pass

    async def _derive_most_valuable(self):
        """ 
        gets the asset with the greatest portion of value stored in it at a step
        in time
        """
        pass

    async def _derive_least_valuable(self):
        """ 
        gets the asset with the smallest portion of value stored in it at a step
        in time
        """
        pass

    # Loss Functions
    # ==========================================================>

    # Balances
    # ==========================================================>

    async def _get_balance(self):
        pass

    async def _set_balance(self):
        pass

    async def _init_zero_balance(self):
        pass

    async def _init_random_balance(self):
        pass

    async def _init_equal_balance(self):
        pass

    # Position
    # ==========================================================>

    async def _close_inactive_positions(self):
        pass

    async def _execute_position(self):
        pass

    async def place_position(self):
        pass



    # Utils
    # ==========================================================>

    async def _random_action(self):
        pass

    async def _step_rate(self):
        pass

    # Metrics
    # ==========================================================>
    