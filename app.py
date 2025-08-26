import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import os
from typing import Optional, List, Dict, Any
import hashlib
import numpy as np


class IncomeMonitorApp:
    def __init__(self, data_file: str = "transactions.json"):
        self.data_file = data_file
        self.transactions = self.load_transactions()

    def load_transactions(self) -> List[Dict]:
        """Wczytaj transakcje z pliku JSON"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('transactions', [])
            except Exception as e:
                st.error(f"Błąd wczytywania danych: {e}")
                return []
        return []

    def save_transactions(self) -> bool:
        """Zapisz transakcje do pliku JSON"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'transactions': self.transactions,
                    'last_updated': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"Błąd zapisywania danych: {e}")
            return False

    def add_transaction(self, date: str, description: str, amount: float,
                        transaction_type: str, tax_rate: float = 0,
                        currency: str = 'PLN', tags: List[str] = None,
                        notes: str = None, investment_amount: float = None) -> bool:
        """Dodaj nową transakcję z podatkiem i typem"""
        try:
            # Oblicz wartości podatkowe
            if transaction_type == "Przychód":
                gross_amount = float(amount)
                tax_amount = gross_amount * (tax_rate / 100)
                net_amount = gross_amount - tax_amount
            else:  # Koszt
                gross_amount = -abs(float(amount))
                tax_amount = 0
                net_amount = gross_amount

            transaction = {
                'id': self.generate_transaction_id(),
                'date': date,
                'description': description,
                'gross_amount': gross_amount,
                'net_amount': net_amount,
                'tax_rate': tax_rate,
                'tax_amount': tax_amount,
                'transaction_type': transaction_type,
                'currency': currency,
                'tags': tags or [],
                'notes': notes or '',
                'investment_amount': investment_amount,
                'created_at': datetime.now().isoformat()
            }

            self.transactions.append(transaction)
            return self.save_transactions()

        except Exception as e:
            st.error(f"Błąd dodawania transakcji: {e}")
            return False

    def generate_transaction_id(self) -> str:
        """Generuj unikalny ID dla transakcji"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:10]

    def delete_transaction(self, transaction_id: str) -> bool:
        """Usuń transakcję"""
        try:
            self.transactions = [t for t in self.transactions if t['id'] != transaction_id]
            return self.save_transactions()
        except Exception as e:
            st.error(f"Błąd usuwania transakcji: {e}")
            return False

    def get_transactions_df(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Pobierz transakcje jako DataFrame z opcjonalnym filtrowaniem po dacie"""
        if not self.transactions:
            return pd.DataFrame()

        df = pd.DataFrame(self.transactions)

        # Konwersja dla starszych transakcji bez nowych pól
        if 'gross_amount' not in df.columns and 'amount' in df.columns:
            df['gross_amount'] = df['amount']
            df['net_amount'] = df['amount']
            df['tax_rate'] = 0
            df['tax_amount'] = 0
            df['transaction_type'] = df['amount'].apply(lambda x: 'Przychód' if x > 0 else 'Koszt')

        df['date'] = pd.to_datetime(df['date'])
        df['gross_amount'] = pd.to_numeric(df['gross_amount'], errors='coerce')
        df['net_amount'] = pd.to_numeric(df['net_amount'], errors='coerce')
        df['tax_amount'] = pd.to_numeric(df['tax_amount'], errors='coerce')
        df['tax_rate'] = pd.to_numeric(df['tax_rate'], errors='coerce')

        # Filtrowanie po dacie
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        return df.sort_values('date', ascending=False)

    def calculate_roi(self, df: pd.DataFrame) -> Dict[str, float]:
        """Oblicz ROI i związane metryki"""
        if df.empty:
            return {
                'roi': 0,
                'total_investment': 0,
                'total_return': 0,
                'profit': 0,
                'roi_percentage': 0
            }

        # Suma kosztów (inwestycji)
        total_investment = abs(df[df['transaction_type'] == 'Koszt']['gross_amount'].sum())

        # Suma przychodów (zwrotu)
        total_return = df[df['transaction_type'] == 'Przychód']['net_amount'].sum()

        # Zysk
        profit = total_return - total_investment

        # ROI
        if total_investment > 0:
            roi_percentage = ((total_return - total_investment) / total_investment) * 100
        else:
            roi_percentage = 0

        return {
            'roi': roi_percentage,
            'total_investment': total_investment,
            'total_return': total_return,
            'profit': profit,
            'roi_percentage': roi_percentage
        }

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Rozszerzone statystyki z uwzględnieniem podatków"""
        if df.empty:
            return {
                'total_income_gross': 0,
                'total_income_net': 0,
                'total_expense': 0,
                'total_tax': 0,
                'net_profit': 0,
                'transaction_count': 0,
                'avg_transaction': 0,
                'effective_tax_rate': 0
            }

        income_df = df[df['transaction_type'] == 'Przychód']
        expense_df = df[df['transaction_type'] == 'Koszt']

        total_income_gross = income_df['gross_amount'].sum()
        total_income_net = income_df['net_amount'].sum()
        total_expense = abs(expense_df['gross_amount'].sum())
        total_tax = income_df['tax_amount'].sum()

        # Efektywna stopa podatkowa
        effective_tax_rate = (total_tax / total_income_gross * 100) if total_income_gross > 0 else 0

        return {
            'total_income_gross': total_income_gross,
            'total_income_net': total_income_net,
            'total_expense': total_expense,
            'total_tax': total_tax,
            'net_profit': total_income_net - total_expense,
            'transaction_count': len(df),
            'avg_transaction': df['gross_amount'].mean(),
            'effective_tax_rate': effective_tax_rate
        }


def create_advanced_dashboard(app: IncomeMonitorApp):
    """Dashboard z zaawansowanymi wykresami i metrykami"""
    st.header("📊 Zaawansowany Dashboard Finansowy")

    # Filtry dat
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Od daty", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("Do daty", value=datetime.now())

    # Pobierz dane
    df = app.get_transactions_df(str(start_date), str(end_date))

    if df.empty:
        st.info("Brak transakcji w wybranym okresie. Dodaj pierwsze transakcje!")
        return

    # Oblicz statystyki i ROI
    stats = app.get_statistics(df)
    roi_metrics = app.calculate_roi(df)

    # SEKCJA 1: Kluczowe metryki KPI
    st.subheader("📈 Kluczowe wskaźniki wydajności (KPI)")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "💰 Przychód brutto",
            f"{stats['total_income_gross']:.2f} PLN",
            delta=f"{stats['total_income_gross'] - stats['total_income_net']:.2f} PLN podatku"
        )

    with col2:
        st.metric(
            "💵 Przychód netto",
            f"{stats['total_income_net']:.2f} PLN",
            delta=f"-{stats['effective_tax_rate']:.1f}% podatku"
        )

    with col3:
        st.metric(
            "💸 Koszty",
            f"{stats['total_expense']:.2f} PLN",
            delta=f"{(stats['total_expense'] / stats['total_income_gross'] * 100) if stats['total_income_gross'] > 0 else 0:.1f}% przychodów"
        )

    with col4:
        roi_color = "normal" if roi_metrics['roi'] > 0 else "inverse"
        st.metric(
            "🎯 ROI",
            f"{roi_metrics['roi']:.1f}%",
            delta=f"Zysk: {roi_metrics['profit']:.2f} PLN",
            delta_color=roi_color
        )

    with col5:
        st.metric(
            "📊 Zysk netto",
            f"{stats['net_profit']:.2f} PLN",
            delta=f"{stats['net_profit'] / stats['total_income_gross'] * 100 if stats['total_income_gross'] > 0 else 0:.1f}% marży"
        )

    # SEKCJA 2: Wykresy główne
    st.subheader("📊 Analiza wizualna")

    tab1, tab2, tab3, tab4 = st.tabs(["Przepływ gotówki", "Struktura", "ROI", "Trendy"])

    with tab1:
        # Wykres przepływu gotówki w czasie
        col1, col2 = st.columns([2, 1])

        with col1:
            daily_flow = df.groupby([df['date'].dt.date, 'transaction_type']).agg({
                'gross_amount': 'sum',
                'net_amount': 'sum'
            }).reset_index()

            fig = go.Figure()

            # Przychody
            income_data = daily_flow[daily_flow['transaction_type'] == 'Przychód']
            fig.add_trace(go.Scatter(
                x=income_data['date'],
                y=income_data['net_amount'],
                mode='lines+markers',
                name='Przychody netto',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,255,0,0.1)'
            ))

            # Koszty
            expense_data = daily_flow[daily_flow['transaction_type'] == 'Koszt']
            fig.add_trace(go.Scatter(
                x=expense_data['date'],
                y=expense_data['gross_amount'],
                mode='lines+markers',
                name='Koszty',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ))

            # Linia zerowa
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

            fig.update_layout(
                title="Przepływ gotówki w czasie",
                xaxis_title="Data",
                yaxis_title="Kwota (PLN)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Wykres kołowy struktury
            pie_data = pd.DataFrame({
                'Typ': ['Przychody netto', 'Koszty', 'Podatki'],
                'Wartość': [stats['total_income_net'], stats['total_expense'], stats['total_tax']]
            })

            fig = px.pie(
                pie_data,
                values='Wartość',
                names='Typ',
                title="Struktura finansowa",
                color_discrete_map={
                    'Przychody netto': '#00CC00',
                    'Koszty': '#FF4444',
                    'Podatki': '#FFA500'
                }
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Analiza struktury przychodów i kosztów
        col1, col2 = st.columns(2)

        with col1:
            # Histogram przychodów
            income_hist = df[df['transaction_type'] == 'Przychód']['gross_amount']
            if not income_hist.empty:
                fig = px.histogram(
                    income_hist,
                    nbins=20,
                    title="Rozkład przychodów",
                    labels={'value': 'Kwota (PLN)', 'count': 'Liczba transakcji'},
                    color_discrete_sequence=['green']
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Histogram kosztów
            expense_hist = df[df['transaction_type'] == 'Koszt']['gross_amount'].abs()
            if not expense_hist.empty:
                fig = px.histogram(
                    expense_hist,
                    nbins=20,
                    title="Rozkład kosztów",
                    labels={'value': 'Kwota (PLN)', 'count': 'Liczba transakcji'},
                    color_discrete_sequence=['red']
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Analiza ROI
        st.subheader("🎯 Analiza zwrotu z inwestycji (ROI)")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Metryki ROI
            st.info(f"""
            **📊 Podsumowanie ROI:**

            💼 **Całkowite inwestycje:** {roi_metrics['total_investment']:.2f} PLN

            💰 **Całkowite zwroty:** {roi_metrics['total_return']:.2f} PLN

            📈 **Zysk:** {roi_metrics['profit']:.2f} PLN

            🎯 **ROI:** {roi_metrics['roi']:.2f}%

            ---

            **Interpretacja:**
            {
            "Świetny wynik! 🎉" if roi_metrics['roi'] > 50 else
            "Dobry wynik! 👍" if roi_metrics['roi'] > 20 else
            "Umiarkowany wynik 📊" if roi_metrics['roi'] > 0 else
            "Strata! ⚠️"
            }
            """)

        with col2:
            # Wykres ROI w czasie (skumulowany)
            df_sorted = df.sort_values('date')
            df_sorted['cumulative_investment'] = df_sorted[df_sorted['transaction_type'] == 'Koszt'][
                'gross_amount'].abs().cumsum()
            df_sorted['cumulative_return'] = df_sorted[df_sorted['transaction_type'] == 'Przychód'][
                'net_amount'].cumsum()

            # Wypełnij NaN wartości
            df_sorted['cumulative_investment'].fillna(method='ffill', inplace=True)
            df_sorted['cumulative_return'].fillna(method='ffill', inplace=True)
            df_sorted.fillna(0, inplace=True)

            # Oblicz ROI dla każdego punktu
            df_sorted['roi_over_time'] = ((df_sorted['cumulative_return'] - df_sorted['cumulative_investment']) /
                                          df_sorted['cumulative_investment'].replace(0, 1)) * 100

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['roi_over_time'],
                mode='lines+markers',
                name='ROI w czasie',
                line=dict(color='purple', width=3),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.1)'
            ))

            # Linia zerowa
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Próg rentowności")

            # Dodaj strefy
            fig.add_hrect(y0=0, y1=df_sorted['roi_over_time'].max() if df_sorted['roi_over_time'].max() > 0 else 100,
                          fillcolor="green", opacity=0.1, annotation_text="Strefa zysku")
            fig.add_hrect(y0=df_sorted['roi_over_time'].min() if df_sorted['roi_over_time'].min() < 0 else -100, y1=0,
                          fillcolor="red", opacity=0.1, annotation_text="Strefa straty")

            fig.update_layout(
                title="Ewolucja ROI w czasie",
                xaxis_title="Data",
                yaxis_title="ROI (%)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Trendy i prognozy
        st.subheader("📈 Analiza trendów")

        # Przygotuj dane miesięczne
        monthly_data = df.groupby([pd.Grouper(key='date', freq='M'), 'transaction_type']).agg({
            'gross_amount': 'sum',
            'net_amount': 'sum'
        }).reset_index()

        # Wykres trendów miesięcznych
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Trendy miesięczne", "Skumulowany bilans"),
            vertical_spacing=0.1
        )

        # Trendy miesięczne
        for trans_type in ['Przychód', 'Koszt']:
            data = monthly_data[monthly_data['transaction_type'] == trans_type]
            if not data.empty:
                fig.add_trace(
                    go.Bar(
                        x=data['date'],
                        y=data['gross_amount'] if trans_type == 'Przychód' else data['gross_amount'].abs(),
                        name=trans_type,
                        marker_color='green' if trans_type == 'Przychód' else 'red'
                    ),
                    row=1, col=1
                )

        # Skumulowany bilans
        df_sorted = df.sort_values('date')
        df_sorted['cumulative_balance'] = df_sorted['net_amount'].cumsum()

        fig.add_trace(
            go.Scatter(
                x=df_sorted['date'],
                y=df_sorted['cumulative_balance'],
                mode='lines',
                name='Bilans skumulowany',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.1)'
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Data", row=2, col=1)
        fig.update_yaxes(title_text="Kwota (PLN)", row=1, col=1)
        fig.update_yaxes(title_text="Bilans (PLN)", row=2, col=1)

        fig.update_layout(height=700, showlegend=True)

        st.plotly_chart(fig, use_container_width=True)

    # SEKCJA 3: Tabela szczegółowa
    st.subheader("📋 Szczegóły transakcji")

    # Filtrowanie
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_type = st.selectbox("Typ transakcji", ["Wszystkie", "Przychód", "Koszt"])
    with col2:
        search_term = st.text_input("🔍 Szukaj w opisie")
    with col3:
        sort_by = st.selectbox("Sortuj według", ["Data", "Kwota", "Podatek"])

    # Aplikuj filtry
    display_df = df.copy()

    if filter_type != "Wszystkie":
        display_df = display_df[display_df['transaction_type'] == filter_type]

    if search_term:
        display_df = display_df[display_df['description'].str.contains(search_term, case=False, na=False)]

    # Sortowanie
    sort_column = {
        "Data": "date",
        "Kwota": "gross_amount",
        "Podatek": "tax_amount"
    }[sort_by]

    display_df = display_df.sort_values(sort_column, ascending=False)

    # Formatowanie do wyświetlenia
    if not display_df.empty:
        display_columns = ['date', 'description', 'transaction_type', 'gross_amount', 'tax_rate', 'tax_amount',
                           'net_amount']
        display_df_formatted = display_df[display_columns].copy()
        display_df_formatted['date'] = display_df_formatted['date'].dt.strftime('%Y-%m-%d')
        display_df_formatted.columns = ['Data', 'Opis', 'Typ', 'Kwota brutto', 'Podatek %', 'Kwota podatku',
                                        'Kwota netto']

        # Kolorowanie według typu
        def color_rows(row):
            if row['Typ'] == 'Przychód':
                return ['background-color: #e6ffe6'] * len(row)
            else:
                return ['background-color: #ffe6e6'] * len(row)

        styled_df = display_df_formatted.style.apply(color_rows, axis=1).format({
            'Kwota brutto': '{:.2f} PLN',
            'Podatek %': '{:.1f}%',
            'Kwota podatku': '{:.2f} PLN',
            'Kwota netto': '{:.2f} PLN'
        })

        st.dataframe(styled_df, use_container_width=True, height=400)


def transaction_form_with_tax(app: IncomeMonitorApp):
    """Formularz dodawania transakcji z podatkiem"""
    st.header("➕ Dodaj nową transakcję")

    # Typ transakcji na górze
    transaction_type = st.radio(
        "Wybierz typ transakcji:",
        ["Przychód", "Koszt"],
        horizontal=True,
        help="Przychód = wpływy, Koszt = wydatki/inwestycje"
    )

    with st.form("transaction_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("📅 Data", value=datetime.now())
            description = st.text_input("📝 Opis", placeholder="np. Faktura za usługi IT")
            amount = st.number_input(
                f"💰 Kwota {'brutto' if transaction_type == 'Przychód' else ''}",
                min_value=0.01,
                step=10.0,
                help=f"Podaj kwotę {'brutto (przed opodatkowaniem)' if transaction_type == 'Przychód' else 'wydatku'}"
            )
            currency = st.selectbox("💱 Waluta", ["PLN", "EUR", "USD", "GBP"])

        with col2:
            # Suwak podatku tylko dla przychodów
            if transaction_type == "Przychód":
                tax_rate = st.slider(
                    "📊 Stawka podatku (%)",
                    min_value=0,
                    max_value=50,
                    value=19,
                    step=1,
                    help="Ustaw stawkę podatku dochodowego"
                )
            else:
                tax_rate = 0

            tags_input = st.text_input("🏷️ Tagi (oddzielone przecinkami)",
                                       placeholder="np. klient1, projekt2")
            tags = [tag.strip() for tag in tags_input.split(',')] if tags_input else []

            notes = st.text_area("📝 Notatki", placeholder="Dodatkowe informacje...")

        # Podsumowanie przed wysłaniem
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            submitted = st.form_submit_button(
                f"{'💰 Dodaj przychód' if transaction_type == 'Przychód' else '💸 Dodaj koszt'}",
                use_container_width=True,
                type="primary"
            )

        with col2:
            if transaction_type == "Przychód":
                # Oblicz net_amount lokalnie tylko do wyświetlenia
                net_amount_display = amount * (1 - tax_rate / 100)
                st.metric("Wpływ na bilans", f"+{net_amount_display:.2f} {currency}")
            else:
                st.metric("Wpływ na bilans", f"-{amount:.2f} {currency}")

        with col3:
            # Oblicz przewidywany wpływ na ROI
            df = app.get_transactions_df()
            current_roi = app.calculate_roi(df)
            st.metric("Obecne ROI", f"{current_roi['roi']:.1f}%")

        if submitted:
            success = app.add_transaction(
                date=str(date),
                description=description,
                amount=amount,
                transaction_type=transaction_type,
                tax_rate=tax_rate if transaction_type == "Przychód" else 0,
                currency=currency,
                tags=tags,
                notes=notes
            )

            if success:
                st.success(f"✅ {transaction_type} został dodany pomyślnie!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Błąd dodawania transakcji")


def manage_transactions_pro(app: IncomeMonitorApp):
    """Zarządzanie transakcjami z zaawansowanymi opcjami"""
    st.header("⚙️ Zarządzanie transakcjami")

    df = app.get_transactions_df()

    if df.empty:
        st.info("Brak transakcji do wyświetlenia. Dodaj pierwszą transakcję!")
        return

    # Panel filtrów
    with st.expander("🔍 Zaawansowane filtry", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            search = st.text_input("Szukaj w opisie", placeholder="Wpisz frazę...")

        with col2:
            transaction_types = ["Wszystkie", "Przychód", "Koszt"]
            selected_type = st.selectbox("Typ transakcji", transaction_types)

        with col3:
            date_range = st.selectbox("Okres",
                                      ["Wszystkie", "Dzisiaj", "Ten tydzień", "Ten miesiąc",
                                       "Ten kwartał", "Ten rok", "Własny zakres"])

        with col4:
            amount_filter = st.selectbox("Filtr kwoty",
                                         ["Wszystkie", "< 100", "100-500", "500-1000",
                                          "1000-5000", "> 5000"])

    # Własny zakres dat
    if date_range == "Własny zakres":
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input("Od daty", value=datetime.now() - timedelta(days=30))
        with col2:
            custom_end = st.date_input("Do daty", value=datetime.now())

    # Aplikuj filtry
    filtered_df = df.copy()

    # Filtr tekstowy
    if search:
        filtered_df = filtered_df[filtered_df['description'].str.contains(search, case=False, na=False)]

    # Filtr typu
    if selected_type != "Wszystkie":
        filtered_df = filtered_df[filtered_df['transaction_type'] == selected_type]

    # Filtr daty
    if date_range == "Dzisiaj":
        filtered_df = filtered_df[filtered_df['date'].dt.date == datetime.now().date()]
    elif date_range == "Ten tydzień":
        filtered_df = filtered_df[filtered_df['date'] >= datetime.now() - timedelta(days=7)]
    elif date_range == "Ten miesiąc":
        filtered_df = filtered_df[filtered_df['date'].dt.month == datetime.now().month]
    elif date_range == "Ten kwartał":
        current_quarter = (datetime.now().month - 1) // 3 + 1
        filtered_df = filtered_df[filtered_df['date'].dt.quarter == current_quarter]
    elif date_range == "Ten rok":
        filtered_df = filtered_df[filtered_df['date'].dt.year == datetime.now().year]
    elif date_range == "Własny zakres":
        filtered_df = filtered_df[(filtered_df['date'].dt.date >= custom_start) &
                                  (filtered_df['date'].dt.date <= custom_end)]

    # Filtr kwoty
    if amount_filter != "Wszystkie":
        abs_amount = filtered_df['gross_amount'].abs()
        if amount_filter == "< 100":
            filtered_df = filtered_df[abs_amount < 100]
        elif amount_filter == "100-500":
            filtered_df = filtered_df[(abs_amount >= 100) & (abs_amount < 500)]
        elif amount_filter == "500-1000":
            filtered_df = filtered_df[(abs_amount >= 500) & (abs_amount < 1000)]
        elif amount_filter == "1000-5000":
            filtered_df = filtered_df[(abs_amount >= 1000) & (abs_amount < 5000)]
        elif amount_filter == "> 5000":
            filtered_df = filtered_df[abs_amount > 5000]

    # Statystyki filtrowanych danych
    if not filtered_df.empty:
        stats = app.get_statistics(filtered_df)
        roi = app.calculate_roi(filtered_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Znalezione transakcje", len(filtered_df))
        with col2:
            st.metric("Suma przychodów", f"{stats['total_income_gross']:.2f} PLN")
        with col3:
            st.metric("Suma kosztów", f"{stats['total_expense']:.2f} PLN")
        with col4:
            st.metric("ROI w okresie", f"{roi['roi']:.1f}%")

        # Tabela transakcji
        st.subheader("📋 Lista transakcji")

        # Przygotuj dane do wyświetlenia
        display_columns = ['date', 'description', 'transaction_type', 'gross_amount',
                           'tax_rate', 'net_amount', 'id']
        display_df = filtered_df[display_columns].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df.columns = ['Data', 'Opis', 'Typ', 'Kwota brutto', 'Podatek %', 'Kwota netto', 'ID']

        # Checkbox do zaznaczania
        display_df.insert(0, 'Zaznacz', False)

        # Edytowalny dataframe
        edited_df = st.data_editor(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Zaznacz": st.column_config.CheckboxColumn(
                    "✓",
                    help="Zaznacz do operacji grupowych",
                    default=False,
                ),
                "Kwota brutto": st.column_config.NumberColumn(
                    "Kwota brutto",
                    format="%.2f PLN",
                ),
                "Podatek %": st.column_config.NumberColumn(
                    "Podatek %",
                    format="%.1f%%",
                ),
                "Kwota netto": st.column_config.NumberColumn(
                    "Kwota netto",
                    format="%.2f PLN",
                ),
                "ID": st.column_config.Column(
                    "ID",
                    disabled=True,
                    width="small"
                )
            },
            disabled=['Data', 'Opis', 'Typ', 'Kwota brutto', 'Podatek %', 'Kwota netto', 'ID']
        )

        # Operacje grupowe
        selected_rows = edited_df[edited_df['Zaznacz'] == True]

        if len(selected_rows) > 0:
            st.subheader(f"🔧 Operacje na {len(selected_rows)} zaznaczonych transakcjach")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🗑️ Usuń zaznaczone", type="secondary", use_container_width=True):
                    deleted_count = 0
                    for _, row in selected_rows.iterrows():
                        if app.delete_transaction(row['ID']):
                            deleted_count += 1

                    if deleted_count > 0:
                        st.success(f"Usunięto {deleted_count} transakcji")
                        st.rerun()

            with col2:
                selected_sum = filtered_df[filtered_df['id'].isin(selected_rows['ID'])]['gross_amount'].sum()
                st.metric("Suma zaznaczonych", f"{selected_sum:.2f} PLN")

            with col3:
                selected_net = filtered_df[filtered_df['id'].isin(selected_rows['ID'])]['net_amount'].sum()
                st.metric("Suma netto zaznaczonych", f"{selected_net:.2f} PLN")
    else:
        st.warning("Brak transakcji spełniających kryteria filtrowania")


def export_import_pro(app: IncomeMonitorApp):
    """Zaawansowany eksport i import z analizą"""
    st.header("📤 Eksport / 📥 Import danych")

    tab1, tab2, tab3 = st.tabs(["Eksport", "Import", "Backup"])

    with tab1:
        st.subheader("📤 Eksport danych")

        df = app.get_transactions_df()

        if not df.empty:
            # Opcje eksportu
            col1, col2 = st.columns(2)

            with col1:
                export_format = st.selectbox("Format eksportu", ["CSV", "Excel", "JSON", "PDF (raport)"])

                date_range = st.selectbox("Zakres danych",
                                          ["Wszystkie", "Ten miesiąc", "Ten kwartał", "Ten rok", "Własny"])

                if date_range == "Własny":
                    start_date = st.date_input("Od", value=datetime.now() - timedelta(days=30))
                    end_date = st.date_input("Do", value=datetime.now())

            with col2:
                include_stats = st.checkbox("Dołącz statystyki", value=True)
                include_charts = st.checkbox("Dołącz wykresy", value=False)
                include_roi = st.checkbox("Dołącz analizę ROI", value=True)

            # Filtruj dane według zakresu
            export_df = df.copy()

            if date_range == "Ten miesiąc":
                export_df = export_df[export_df['date'].dt.month == datetime.now().month]
            elif date_range == "Ten kwartał":
                current_quarter = (datetime.now().month - 1) // 3 + 1
                export_df = export_df[export_df['date'].dt.quarter == current_quarter]
            elif date_range == "Ten rok":
                export_df = export_df[export_df['date'].dt.year == datetime.now().year]
            elif date_range == "Własny":
                export_df = export_df[(export_df['date'] >= pd.to_datetime(start_date)) &
                                      (export_df['date'] <= pd.to_datetime(end_date))]

            # Przygotuj dane
            export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')

            # Generuj eksport
            if st.button("🚀 Generuj eksport", type="primary", use_container_width=True):

                if export_format == "CSV":
                    csv = export_df.to_csv(index=False).encode('utf-8')

                    # Dodaj statystyki jeśli wybrano
                    if include_stats:
                        stats = app.get_statistics(df)
                        roi = app.calculate_roi(df)

                        stats_text = f"\n\n# STATYSTYKI\n"
                        stats_text += f"Przychód brutto,{stats['total_income_gross']:.2f}\n"
                        stats_text += f"Przychód netto,{stats['total_income_net']:.2f}\n"
                        stats_text += f"Koszty,{stats['total_expense']:.2f}\n"
                        stats_text += f"Podatek,{stats['total_tax']:.2f}\n"
                        stats_text += f"Zysk netto,{stats['net_profit']:.2f}\n"
                        stats_text += f"ROI,{roi['roi']:.2f}%\n"

                        csv = csv + stats_text.encode('utf-8')

                    st.download_button(
                        label="⬇️ Pobierz CSV",
                        data=csv,
                        file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )

                elif export_format == "JSON":
                    export_data = {
                        'metadata': {
                            'export_date': datetime.now().isoformat(),
                            'transaction_count': len(export_df),
                            'date_range': date_range
                        },
                        'transactions': export_df.to_dict('records')
                    }

                    if include_stats:
                        stats = app.get_statistics(df)
                        roi = app.calculate_roi(df)
                        export_data['statistics'] = {
                            'total_income_gross': stats['total_income_gross'],
                            'total_income_net': stats['total_income_net'],
                            'total_expense': stats['total_expense'],
                            'total_tax': stats['total_tax'],
                            'net_profit': stats['net_profit'],
                            'roi': roi['roi']
                        }

                    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

                    st.download_button(
                        label="⬇️ Pobierz JSON",
                        data=json_str,
                        file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json'
                    )

                st.success("✅ Eksport wygenerowany pomyślnie!")
        else:
            st.info("Brak danych do eksportu")

    with tab2:
        st.subheader("📥 Import danych")

        uploaded_file = st.file_uploader(
            "Wybierz plik do importu",
            type=['csv', 'json', 'xlsx'],
            help="Obsługiwane formaty: CSV, JSON, Excel"
        )

        if uploaded_file is not None:
            # Podgląd danych
            st.subheader("👀 Podgląd danych")

            try:
                if uploaded_file.name.endswith('.csv'):
                    import_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    import_data = json.load(uploaded_file)
                    if isinstance(import_data, dict) and 'transactions' in import_data:
                        import_df = pd.DataFrame(import_data['transactions'])
                    else:
                        import_df = pd.DataFrame(import_data)
                elif uploaded_file.name.endswith('.xlsx'):
                    import_df = pd.read_excel(uploaded_file)

                st.dataframe(import_df.head(10), use_container_width=True)
                st.info(f"Znaleziono {len(import_df)} transakcji do importu")

                # Mapowanie kolumn
                st.subheader("🔄 Mapowanie kolumn")

                col1, col2 = st.columns(2)

                with col1:
                    date_col = st.selectbox("Kolumna z datą", import_df.columns,
                                            index=0 if 'date' in import_df.columns else 0)
                    desc_col = st.selectbox("Kolumna z opisem", import_df.columns,
                                            index=1 if 'description' in import_df.columns else 1)
                    amount_col = st.selectbox("Kolumna z kwotą", import_df.columns,
                                              index=2 if 'amount' in import_df.columns else 2)

                with col2:
                    type_col = st.selectbox("Kolumna z typem (opcjonalnie)",
                                            ["Brak"] + list(import_df.columns))
                    tax_col = st.selectbox("Kolumna z podatkiem (opcjonalnie)",
                                           ["Brak"] + list(import_df.columns))

                    default_tax = st.slider("Domyślny podatek (%)", 0, 50, 19)

                # Import
                if st.button("📥 Importuj dane", type="primary", use_container_width=True):
                    success_count = 0
                    error_count = 0

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, row in import_df.iterrows():
                        try:
                            # Określ typ transakcji
                            if type_col != "Brak" and type_col in row:
                                trans_type = row[type_col]
                            else:
                                # Automatyczne wykrywanie na podstawie kwoty
                                trans_type = "Przychód" if float(row[amount_col]) > 0 else "Koszt"

                            # Określ podatek
                            if tax_col != "Brak" and tax_col in row:
                                tax_rate = float(row[tax_col])
                            else:
                                tax_rate = default_tax if trans_type == "Przychód" else 0

                            # Dodaj transakcję
                            success = app.add_transaction(
                                date=str(row[date_col]),
                                description=str(row[desc_col]),
                                amount=abs(float(row[amount_col])),
                                transaction_type=trans_type,
                                tax_rate=tax_rate,
                                currency="PLN"
                            )

                            if success:
                                success_count += 1
                            else:
                                error_count += 1

                            # Aktualizuj progress
                            progress = (idx + 1) / len(import_df)
                            progress_bar.progress(progress)
                            status_text.text(f"Importowanie: {idx + 1}/{len(import_df)}")

                        except Exception as e:
                            error_count += 1
                            st.warning(f"Błąd w wierszu {idx + 1}: {str(e)}")

                    progress_bar.empty()
                    status_text.empty()

                    st.success(f"""
                    ✅ Import zakończony!
                    - Zaimportowano: {success_count} transakcji
                    - Błędy: {error_count} transakcji
                    """)

                    if success_count > 0:
                        st.balloons()
                        st.rerun()

            except Exception as e:
                st.error(f"Błąd wczytywania pliku: {str(e)}")

    with tab3:
        st.subheader("💾 Backup i przywracanie")

        col1, col2 = st.columns(2)

        with col1:
            st.info("**Utwórz backup**")

            backup_name = st.text_input("Nazwa backupu",
                                        value=f"backup_{datetime.now().strftime('%Y%m%d')}")

            if st.button("💾 Utwórz pełny backup", use_container_width=True):
                # Przygotuj pełny backup
                df = app.get_transactions_df()
                stats = app.get_statistics(df)
                roi = app.calculate_roi(df)

                backup_data = {
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'name': backup_name,
                    'statistics': {
                        'transaction_count': len(df),
                        'total_income_gross': stats['total_income_gross'],
                        'total_income_net': stats['total_income_net'],
                        'total_expense': stats['total_expense'],
                        'net_profit': stats['net_profit'],
                        'roi': roi['roi']
                    },
                    'transactions': app.transactions
                }

                backup_json = json.dumps(backup_data, ensure_ascii=False, indent=2)

                st.download_button(
                    label="⬇️ Pobierz backup",
                    data=backup_json,
                    file_name=f"{backup_name}.json",
                    mime='application/json'
                )

                st.success("✅ Backup utworzony pomyślnie!")

        with col2:
            st.info("**Przywróć z backupu**")

            backup_file = st.file_uploader("Wybierz plik backupu", type=['json'])

            if backup_file is not None:
                try:
                    backup_data = json.load(backup_file)

                    st.write(f"**Informacje o backupie:**")
                    st.write(f"- Nazwa: {backup_data.get('name', 'Nieznana')}")
                    st.write(f"- Data utworzenia: {backup_data.get('created_at', 'Nieznana')}")
                    st.write(f"- Liczba transakcji: {len(backup_data.get('transactions', []))}")

                    if st.button("♻️ Przywróć backup", type="primary", use_container_width=True):
                        # Zastąp dane
                        app.transactions = backup_data['transactions']
                        if app.save_transactions():
                            st.success("✅ Backup przywrócony pomyślnie!")
                            st.rerun()
                        else:
                            st.error("❌ Błąd przywracania backupu")

                except Exception as e:
                    st.error(f"Błąd odczytu backupu: {str(e)}")


def main():
    st.set_page_config(
        page_title="Income Monitor PRO",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("💰 Income Monitor PRO - Zaawansowany System Finansowy")
    st.caption("Wersja 2.0 - z analizą podatków i ROI")

    # Inicjalizacja aplikacji
    app = IncomeMonitorApp()

    # Menu boczne
    with st.sidebar:
        st.header("📱 Menu główne")

        page = st.radio(
            "Wybierz moduł",
            ["🏠 Dashboard", "➕ Nowa transakcja", "⚙️ Zarządzanie", "📤 Import/Export"],
            label_visibility="collapsed"
        )

        st.divider()

        # Widget szybkich statystyk
        df = app.get_transactions_df()
        if not df.empty:
            stats = app.get_statistics(df)
            roi = app.calculate_roi(df)

            st.subheader("📊 Szybki podgląd")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Bilans netto", f"{stats['net_profit']:.0f} PLN")
                st.metric("ROI", f"{roi['roi']:.1f}%")
            with col2:
                st.metric("Efektywny podatek", f"{stats['effective_tax_rate']:.1f}%")
                st.metric("Transakcji", stats['transaction_count'])

        st.divider()

        # Ustawienia
        with st.expander("⚙️ Ustawienia"):
            theme = st.selectbox("Motyw", ["Jasny", "Ciemny", "Auto"])
            currency = st.selectbox("Domyślna waluta", ["PLN", "EUR", "USD"])
            tax_rate = st.slider("Domyślny podatek (%)", 0, 50, 19)

            if st.button("💾 Zapisz ustawienia"):
                st.success("Ustawienia zapisane!")

        st.divider()

        # Informacje
        st.info(
            """
            **Income Monitor PRO v2.0**

            System zarządzania finansami z analizą ROI i podatków.

            © 2024 - Wszystkie prawa zastrzeżone
            """
        )

    # Routing stron
    if page == "🏠 Dashboard":
        create_advanced_dashboard(app)
    elif page == "➕ Nowa transakcja":
        transaction_form_with_tax(app)
    elif page == "⚙️ Zarządzanie":
        manage_transactions_pro(app)
    elif page == "📤 Import/Export":
        export_import_pro(app)


if __name__ == "__main__":
    main()