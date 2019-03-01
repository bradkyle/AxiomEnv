from enum import Enum

class SandboxEnv():
    def __init__(
        self,
        buffer,
        quote_asset,
        commission,
        asset_num,
        window_size,
        feature_num,
        selection_period,
        selection_method,
        balance_init
        ):
        
        self.buffer=buffer
        self.quote_asset = quote_asset
        self.commission = commission
        self.asset_num = asset_num
        self.window_size = window_size
        self.feature_num = feature_num
        self.selection_period = selection_period
        self.selection_method = selection_method
        self.balance_init = balance_init
        self.balances = {}

        self.prev_tnorm=0
        self.stepped=False
        self.step_count=0
        self.prev_action_pv = [0]

    async def step(self, action):
        feature_frame, assets = await self.buffer.get_state(
            asset_num=self.asset_num,
            window_size=self.window_size,
            feature_num=self.feature_num,
            selection_period=self.selection_period,
            selection_method=self.selection_method
        )

        await self.close_inactive_positions(
            quote_asset=self.quote_asset,
            assets=assets
        )

        current_pv = await self.derive_pv(
            quote_asset=self.quote_asset,
            assets=assets
        )

        await self.execute_position(
            p_vector=action,
            assets=assets,
            quote_asset=self.quote_asset
        )

        tnorm = await self.derive_tnorm(
            quote_asset=self.quote_asset
        )

        self.prev_tnorm = tnorm
        self.stepped = True
        self.step_count +=1
        self.prev_action_pv = action

        profit = tnorm - self.prev_tnorm

        return [
            assets,
            feature_frame, 
            current_pv, 
            tnorm,
            profit
        ]

    async def state(self):
        feature_frame, assets = await self.buffer.get_state(
            asset_num=self.asset_num,
            window_size=self.window_size,
            feature_num=self.feature_num,
            selection_period=self.selection_period,
            selection_method=self.selection_method
        )

        current_pv = await self.derive_pv(
            quote_asset=self.quote_asset,
            assets=assets
        )

        pv_prices, pv_values = await self.derive_current_pv_info(
            quote_asset=self.quote_asset,
            assets=assets,
            current_pv=current_pv
        )

        tnorm = await self.derive_tnorm(
            quote_asset=self.quote_asset
        )

        return [
            assets, 
            feature_frame, 
            current_pv, 
            pv_prices, 
            pv_values, 
            tnorm
        ]

    async def act(self, action):
        raise NotImplemented

    async def reset(self):
        feature_frame, assets = await self.buffer.get_state(
            asset_num=self.asset_num,
            window_size=self.window_size,
            feature_num=self.feature_num,
            selection_period=self.selection_period,
            selection_method=self.selection_method
        )

        await self.init_balance(assets)

        pv = await self.derive_pv(
            self.quote_asset,
            assets
        )

        return [
            pv, 
            feature_frame, 
            assets
        ] 

    async def close(self):
        pass


    # 
    #======================================================================>

    async def derive_pv(self, quote_asset, assets):
        current_pv = []
        tnorm = await self.derive_tnorm(quote_asset)
        for i, asset in enumerate(assets):
            last_price = await self.buffer.get_last_price(asset+quote_asset)
            balance = self.balances[asset]
            norm_balance = balance * last_price
            if tnorm>0:
                p_scalar = norm_balance/tnorm
            else:
                p_scalar = 0
            current_pv.append(round(p_scalar, 3))
        return current_pv

    # TODO make async
    async def derive_tnorm(self, quote_asset):
        tnorm = 0
        for asset, balance in self.balances.items():
            if balance is not None:
                if balance>0:
                    if asset != quote_asset:
                        last_price = await self.buffer.get_last_price(asset+quote_asset)
                    else:
                        last_price = 1
                    value = balance * last_price
                    tnorm += value
        return tnorm

    # TODO fix and make async
    async def derive_current_pv_info(self, quote_asset, assets, current_pv):
        pv_values = []
        pv_prices = []
        for i, asset in enumerate(assets):
            last_price = await self.buffer.get_last_price(asset+quote_asset)
            value = current_pv[i] 
            pv_values.append(value)
            pv_prices.append(last_price)
        return pv_prices, pv_values

    async def init_balance(self, assets):
        await self.init_basic_balances(
            self.quote_asset,
            1,
            assets
        )

    async def init_zero_balances(self, assets):
        pass


    # TODO allow for randomization of starting balance
    async def init_basic_balances(self, quote_asset, starting_balance, assets):
        self.balances[quote_asset] = starting_balance
        for asset in assets:
            self.balances[asset] = 0 


    async def init_randomized_balance(self):
        pass


    # TODO make async
    async def close_inactive_positions(self, quote_asset, assets):
        for asset, balance in self.balances.items():
            if not asset in assets:
                if await self.buffer.has_symbol_info(asset+quote_asset):
                    await self.create_virtual_trade(
                        asset,
                        quote_asset,
                        balance,
                        Side.SELL,
                        False
                    )
                else:
                    raise NotImplementedError(
                        "Symbol does not have info:" + asset+quote_asset
                    )


    # TODO make async
    async def execute_position(self, p_vector, assets, quote_asset):
        print(p_vector)

        for i, p_scalar in enumerate(p_vector):
            await self.place_position(
                assets[i],
                quote_asset,
                p_scalar,
                Side.SELL
            )

        for i, p_scalar in enumerate(p_vector):
            await self.place_position(
                assets[i],
                quote_asset,
                p_scalar,
                Side.BUY
            )


    async def place_position(self, base_asset, quote_asset, p_scalar, side):
        tnorm = await self.derive_tnorm(quote_asset)

        last_price = await self.buffer.get_last_price(base_asset+quote_asset)
        balance = await self.get_balance(base_asset)
        current_norm = balance * last_price

        target_norm = p_scalar * tnorm

        if current_norm > target_norm and side == Side.SELL:
            norm_quantity = current_norm - target_norm
            await self.create_virtual_trade()
        elif current_norm < target_norm and side == Side.BUY:
            norm_quantity = current_norm - target_norm
            await self.create_virtual_trade()
        else:
            pass


    async def create_virtual_trade(self, base_asset, quote_asset, quantity, side, is_norm=True):
        best_price = await self.buffer.get_last_price(base_asset+quote_asset)

        quote_balance = await self.get_balance(quote_asset)
        base_balance = await self.get_balance(base_asset)

        # todo make sure enough balanace exists
        if is_norm:
            if side == Side.BUY:
                new_quote_balance = quote_balance - quantity
                base_quantity = quantity/best_price
                new_base_balance = base_balance + (base_quantity - (base_quantity*self.commission))

            elif side == Side.SELL:
                new_quote_balance = quote_balance + (quantity - (quantity*self.commission))
                new_base_balance = base_balance - (quote_balance/best_price)
        else:
            if side == Side.BUY:
                quote_quantity = (quantity*best_price)
                new_quote_balance = quote_balance - quote_quantity
                new_base_balance = base_balance + (quantity - (quantity*self.commission))

            elif side == Side.SELL:
                quote_quantity = (quantity*best_price)
                new_quote_balance = quote_balance + (quote_quantity - (quote_quantity*self.commission))
                new_base_balance = base_balance - quantity
        
        await self.update_balance(quote_asset, round(new_quote_balance, 9))
        await self.update_balance(base_asset, round(new_base_balance, 9))
                

    async def update_balance(self, asset, balance):
        if asset in self.balances:
            self.balances[asset] = balance
        else:
            raise KeyError("Balance for given asset does not exist")

    async def get_balance(self, asset):
        if asset in self.balances:
            return self.balances[asset];
        else:
            raise KeyError("Balance for given asset does not exist")


    
class Side(Enum):
     BUY = 1
     SELL = 2