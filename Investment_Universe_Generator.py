import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string


class BondDataGenerator:
    """
    Outil pour generer un fichier excel avec tous les parametres d'un indice Bloomberg
    """

    def __init__(self, num_bonds=1000, random_state=42):
        """
        Definition des parametres a generer
        """
        self.num_bonds = num_bonds
        np.random.seed(random_state)
        random.seed(random_state)

        self.regions = [
            'North America', 'Europe', 'Asia', 'Latin America', 'Middle East', 'Africa'
        ]

        self.countries = {
            'North America': ['United States', 'Canada', 'Mexico'],
            'Europe': ['Germany', 'France', 'United Kingdom', 'Italy', 'Spain', 'Netherlands', 'Switzerland','Turkey', 'Poland'],
            'Asia': ['Japan', 'Australia', 'Singapore', 'South Korea', 'Hong Kong','India', 'China', 'Indonesia'],
            'Latin America': ['Brazil', 'Chile', 'Colombia', 'Peru'],
            'Middle East': ['Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Israel'],
            'Africa': ['South Africa', 'Egypt', 'Nigeria']
        }

        self.sectors = [
            'Government', 'Corporate', 'Financial', 'Utility', 'Industrial',
            'Energy', 'Technology', 'Healthcare', 'Consumer', 'Telecommunications'
        ]

        self.ratings = [
            'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
            'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-'
        ]

        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']

    def _generate_isin(self, country_code, index):
        """ On s'amuse a generer un code ISIN realiste, aucun interet ici si ce n'est pour lire les resultats sans
        output les noms des bonds"""
        random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
        check_digit = random.randint(0, 9)
        return f"{country_code}{random_part}{check_digit}"

    def _get_country_code(self, country):
        """ Bien long """
        codes = {
            'United States': 'US', 'Canada': 'CA', 'Mexico': 'MX',
            'Germany': 'DE', 'France': 'FR', 'United Kingdom': 'GB',
            'Italy': 'IT', 'Spain': 'ES', 'Netherlands': 'NL',
            'Switzerland': 'CH', 'Japan': 'JP', 'Australia': 'AU',
            'Singapore': 'SG', 'South Korea': 'KR', 'Hong Kong': 'HK',
            'Brazil': 'BR', 'Chile': 'CL', 'Colombia': 'CO', 'Peru': 'PE',
            'Saudi Arabia': 'SA', 'UAE': 'AE', 'Qatar': 'QA', 'Kuwait': 'KW',
            'South Africa': 'ZA', 'Egypt': 'EG', 'Nigeria': 'NG',
            'India': 'IN', 'China': 'CN', 'Indonesia': 'ID',
            'Turkey': 'TR', 'Poland': 'PL'
        }
        return codes.get(country, 'XX')

    def _calculate_ytm_from_rating(self, rating, maturity_years):
        """ On donne un yield par note"""
        base_rates = {
            'AAA': 2.5, 'AA+': 2.7, 'AA': 2.9, 'AA-': 3.1,
            'A+': 3.3, 'A': 3.5, 'A-': 3.8,
            'BBB+': 4.2, 'BBB': 4.5, 'BBB-': 5.0,
            'BB+': 5.8, 'BB': 6.5, 'BB-': 7.2
        }

        base_rate = base_rates[rating]

        # Ajout du term premium
        if maturity_years < 2:
            term_premium = -0.3
        elif maturity_years < 5:
            term_premium = 0.2
        elif maturity_years < 10:
            term_premium = 0.5
        elif maturity_years < 20:
            term_premium = 0.8
        else:
            term_premium = 1.0

        # Poudre de perlinpinpin stochastique dans le YTM
        noise = np.random.normal(0, 0.3)

        return max(0.5, base_rate + term_premium + noise)

    def _calculate_duration(self, maturity_years, ytm):
        """on input la modified duration : le mieux aurait ete de calculer la key rate duration mais par
        simplification on ajoute juste un bruit """
        macaulay_duration = 0.75 * maturity_years
        modified_duration = macaulay_duration / (1 + ytm / 100)

        variation = np.random.normal(1, 0.05)
        return max(0.1, modified_duration * variation)

    def generate_bonds(self):
        """ on genere les x obligations"""
        bonds = []

        region_weights = np.random.dirichlet(np.ones(len(self.regions)) * 2)
        region_distribution = dict(zip(self.regions, region_weights))

        for i in range(self.num_bonds):
            region = np.random.choice(self.regions, p=region_weights)

            country = random.choice(self.countries[region])
            country_code = self._get_country_code(country)

            isin = self._generate_isin(country_code, i)

            sector = random.choice(self.sectors)
            rating = random.choice(self.ratings)

            maturity_years = np.random.uniform(0.5, 30)

            days_ago = random.randint(0, 3650)
            issue_date = datetime.now() - timedelta(days=days_ago)

            maturity_date = issue_date + timedelta(days=int(maturity_years * 365))

            ytm = self._calculate_ytm_from_rating(rating, maturity_years)

            duration = self._calculate_duration(maturity_years, ytm)

            coupon = ytm + np.random.normal(0, 0.5)
            coupon = max(0.5, min(15, coupon))  # Cap between 0.5% and 15%

            if region == 'North America':
                currency = random.choices(['USD', 'CAD'], weights=[0.8, 0.2])[0]
            elif region == 'Europe':
                currency = random.choices(['EUR', 'GBP', 'CHF'], weights=[0.6, 0.3, 0.1])[0]
            elif region == 'Asia Pacific':
                currency = random.choices(['USD', 'JPY', 'AUD'], weights=[0.5, 0.3, 0.2])[0]
            else:
                currency = random.choices(['USD', 'EUR'], weights=[0.7, 0.3])[0]

            outstanding = np.random.lognormal(mean=5, sigma=1.5) * 10
            outstanding = round(outstanding, 2)

            price = 100 + np.random.normal(0, 5)
            price = max(70, min(130, price))

            bond = {
                'ISIN': isin,
                'Country': country,
                'Region': region,
                'Sector': sector,
                'Rating': rating,
                'Currency': currency,
                'Issue_Date': issue_date.strftime('%Y-%m-%d'),
                'Maturity_Date': maturity_date.strftime('%Y-%m-%d'),
                'MTY_YEARS': round(maturity_years, 4),
                'Coupon': round(coupon, 4),
                'YLD_YTM_MID': round(ytm, 4),
                'DUR_ADJ_MID': round(duration, 4),
                'Price': round(price, 4),
                'Outstanding_Amount_M': round(outstanding, 2)
            }

            bonds.append(bond)

        df = pd.DataFrame(bonds)
        df['Market_Value'] = df['Outstanding_Amount_M'] * df['Price'] / 100
        total_market_value = df['Market_Value'].sum()
        df['Benchmark_Weight'] = df['Market_Value'] / total_market_value
        df['w ytm'] = df['Benchmark_Weight'] * df['YLD_YTM_MID']
        df['w dur'] = df['Benchmark_Weight'] * df['DUR_ADJ_MID']
        df['w maturity'] = df['Benchmark_Weight'] * df['MTY_YEARS']

        return df

    def add_statistics_sheet(self, df):
        """ On ajoute une sheet synthese"""
        stats = {
            'Metric': [],
            'Value': []
        }

        stats['Metric'].extend([
            'Total Bonds',
            'Total Market Value (M)',
            'Average YTM (%)',
            'Weighted Average YTM (%)',
            'Average Duration',
            'Weighted Average Duration',
            'Average Maturity (Years)',
            'Weighted Average Maturity (Years)',
            '',
            'By Region'
        ])

        stats['Value'].extend([
            len(df),
            f"{df['Market_Value'].sum():,.2f}",
            f"{df['YLD_YTM_MID'].mean():.4f}",
            f"{df['w ytm'].sum():.4f}",
            f"{df['DUR_ADJ_MID'].mean():.4f}",
            f"{df['w dur'].sum():.4f}",
            f"{df['MTY_YEARS'].mean():.4f}",
            f"{df['w maturity'].sum():.4f}",
            '',
            ''
        ])

        region_dist = df.groupby('Region').agg({
            'ISIN': 'count',
            'Benchmark_Weight': 'sum',
            'Market_Value': 'sum'
        }).round(4)

        for region in region_dist.index:
            stats['Metric'].append(f"  {region}")
            stats['Value'].append(f"{region_dist.loc[region, 'Benchmark_Weight'] * 100:.2f}%")

        stats['Metric'].append('')
        stats['Value'].append('')
        stats['Metric'].append('By Rating')
        stats['Value'].append('')

        rating_dist = df.groupby('Rating').agg({
            'ISIN': 'count',
            'Benchmark_Weight': 'sum'
        }).round(4)

        for rating in rating_dist.index:
            stats['Metric'].append(f"  {rating}")
            stats['Value'].append(f"{rating_dist.loc[rating, 'Benchmark_Weight'] * 100:.2f}%")

        return pd.DataFrame(stats)


if __name__ == "__main__":
    print("*" * 80)
    print("Generation de l'excel")
    print("*" * 80)

    generator = BondDataGenerator(num_bonds=1000, random_state=42)
    bonds_df = generator.generate_bonds()
    stats_df = generator.add_statistics_sheet(bonds_df)

    print("\n" + "*" * 80)
    print("check sur les 10 premiers")
    print("*" * 80)
    print(bonds_df.head(10).to_string(index=False))

    print("\n" + "*" * 80)
    print("Stats")
    print("*" * 80)
    print(stats_df.to_string(index=False))

    excel_filename = 'synthetic_bond_data.xlsx'

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        bonds_df.to_excel(writer, sheet_name='Bond_Data', index=False)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        regional_summary = bonds_df.groupby('Region').agg({
            'ISIN': 'count',
            'Market_Value': 'sum',
            'Benchmark_Weight': 'sum',
            'YLD_YTM_MID': 'mean',
            'DUR_ADJ_MID': 'mean',
            'MTY_YEARS': 'mean'
        }).round(4)
        regional_summary.columns = ['Count', 'Market_Value_M', 'Total_Weight', 'Avg_YTM', 'Avg_Duration',
                                    'Avg_Maturity']
        regional_summary.to_excel(writer, sheet_name='Regional_Summary')

        rating_summary = bonds_df.groupby('Rating').agg({
            'ISIN': 'count',
            'Market_Value': 'sum',
            'Benchmark_Weight': 'sum',
            'YLD_YTM_MID': 'mean',
            'DUR_ADJ_MID': 'mean'
        }).round(4)
        rating_summary.columns = ['Count', 'Market_Value_M', 'Total_Weight', 'Avg_YTM', 'Avg_Duration']
        rating_summary.to_excel(writer, sheet_name='Rating_Summary')

        # Country breakdown
        country_summary = bonds_df.groupby('Country').agg({
            'ISIN': 'count',
            'Market_Value': 'sum',
            'Benchmark_Weight': 'sum'
        }).round(4).sort_values('Market_Value', ascending=False)
        country_summary.columns = ['Count', 'Market_Value_M', 'Total_Weight']
        country_summary.to_excel(writer, sheet_name='Country_Summary')


    csv_filename = 'synthetic_bond_data.csv'
    bonds_df.to_csv(csv_filename, index=False)