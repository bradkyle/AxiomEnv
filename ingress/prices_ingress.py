            price = {
                fields.ID:k['s'],
                fields.EVENT_TIME: d['E'],
                fields.EPOCH_EVENT_TIME: r.epoch_time(d['E']/1000),
                fields.QUOTE_ASSET: self.quote_asset,
                fields.BASE_ASSET: self.derive_base_asset(self.quote_asset,k['s']),
                fields.SYMBOL: k['s'],
                fields.PRICE: float(k['c'])
            }
            r.table(db_const.PRICES_TABLE).insert(
                price,
                conflict="replace"
            ).run(self.conn)