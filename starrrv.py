#!/usr/bin/env python
# coding: utf-8

# # Voting System Simulation: STAR + RRV (Fully Vectorized)
#
# - STAR (district) + RRV (national)
# - Tactical alignments:
#   - Party A is index 0.  Parties with even indexes align with Party A
#   - Party B is index 1.  Parties with odd indexes align with Part B
# - National 2% threshold for party eligibility
# - Comparison with FPTP and MMP

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
score_scale = 6

""" Voting Systems """
# STAR voting (district)
def star_voting_winner(score_df):
    totals = score_df.sum()
    if len(totals) < 2:
        return totals.idxmax()
    top_two = totals.nlargest(2).index.tolist()
    runoff = score_df[top_two]
    # Count how many voters prefer the first candidate over the second
    preference_count = (runoff[top_two[0]] > runoff[top_two[1]]).sum()
    # If more than half prefer the first, they win. Otherwise, the second wins.
    if preference_count > len(runoff) / 2:
        return top_two[0]
    return top_two[1]


# Reweighted Range Voting (RRV)
def reweighted_range_voting(score_df, seats):
    if score_df.empty or seats == 0:
        return pd.Series(0, index=score_df.columns)
    scores_np = score_df.to_numpy()
    weights = np.ones(len(scores_np))
    seats_won = pd.Series(0, index=score_df.columns)
    max_score = score_scale - 1

    for _ in range(seats):
        weighted_scores = scores_np * weights[:, np.newaxis]
        winner_idx = weighted_scores.sum(axis=0).argmax()
        winner_party = score_df.columns[winner_idx]
        seats_won[winner_party] += 1

        support = scores_np[:, winner_idx] / max_score
        weights /= (1 + support)
    return seats_won


# FPTP voting
def fptp_winner(score_df):
    # idxmax(axis=1) finds the column name with the max value for each row.
    return score_df.idxmax(axis=1).value_counts().idxmax()


# Tactical voting (Vectorized)
def tactical_voting_vectorized(df, party_names):
    scores = df.to_numpy()
    max_indices = np.argmax(scores, axis=1)

    # Masks for each condition
    mask_is_A = (max_indices == 0)
    mask_is_B = (max_indices == 1)
    mask_is_minor_A = (max_indices >= 2) & (max_indices % 2 == 0)
    mask_is_minor_B = (max_indices >= 2) & (max_indices % 2 != 0)

    # Apply tactical boosts
    scores[mask_is_A, 1] += 1
    scores[mask_is_B, 0] += 1
    scores[mask_is_minor_A, 0] += 1
    scores[mask_is_minor_B, 1] += 1

    # Ensure scores do not exceed the max score
    np.clip(scores, 0, score_scale - 1, out=scores)

    return pd.DataFrame(scores, columns=party_names)


# Threshold filtering
def threshold_filter(df, threshold=0.02):
    if df.empty:
        return df
    totals = df.sum()
    share = totals / totals.sum()
    return df.loc[:, share >= threshold]


# Generate split block scores (Vectorized)
def generate_block_scores_vectorized(v, pt, rng, dominant, favor_minor):
    if v == 0:
        return np.array([]).reshape(0, pt)

    scores = np.zeros((v, pt), dtype=int)

    if favor_minor:
        # Voters who prefer a minor party's coalition
        scores = rng.choice([0, 1], size=(v, pt))
        coalition_A_pref = rng.choice([True, False], size=v)

        # Voters preferring coalition A give high scores to even minor parties
        even_minors = np.arange(2, pt, 2)
        if even_minors.size > 0:
            high_scores_A = rng.choice([3, 4, 5], size=(np.sum(coalition_A_pref), len(even_minors)))
            scores[coalition_A_pref, dominant[0]] = 1
            scores[np.ix_(coalition_A_pref, even_minors)] = high_scores_A

        # Voters preferring coalition B give high scores to odd minor parties
        odd_minors = np.arange(3, pt, 2)
        if odd_minors.size > 0:
            high_scores_B = rng.choice([3, 4, 5], size=(np.sum(~coalition_A_pref), len(odd_minors)))
            scores[~coalition_A_pref, dominant[1]] = 1
            scores[np.ix_(~coalition_A_pref, odd_minors)] = high_scores_B

    else:
        # Loyal major party supporters
        voter_choice = rng.choice(dominant, size=v)
        scores = rng.choice([0, 1], size=(v, pt))

        mask_A = (voter_choice == dominant[0])
        scores[mask_A, dominant[0]] = 5

        mask_B = (voter_choice == dominant[1])
        scores[mask_B, dominant[1]] = 5

    return scores

def generate_split_voter_blocks(n_voters, dominant, rng, ratio, num_parties):
    b1_voters = round(n_voters * ratio)
    b2_voters = n_voters - b1_voters
    block1 = generate_block_scores_vectorized(b1_voters, num_parties, rng, dominant, favor_minor=True)
    block2 = generate_block_scores_vectorized(b2_voters, num_parties, rng, dominant, favor_minor=False)
    return np.vstack((block1, block2)) if b1_voters > 0 and b2_voters > 0 else (block1 if b1_voters > 0 else block2)


def generate_scores_fptp(n, p, dominant, rng, total_domin_pref):
    district_swing = (rng.random() - 0.5) * 0.1
    district_major_pref = np.clip(total_domin_pref + district_swing, 0, 1)
    total_major_voters = round(n * district_major_pref)

    split_swing = (rng.random() - 0.5) * 0.05
    party_A_share = 0.5 + split_swing

    voters_A = round(total_major_voters * party_A_share)
    voters_B = total_major_voters - voters_A
    voters_other = n - voters_A - voters_B

    scores = rng.integers(0, 3, size=(n, p))

    voter_indices = np.arange(n)
    rng.shuffle(voter_indices)

    scores[voter_indices[:voters_A], dominant[0]] = 5
    scores[voter_indices[voters_A:voters_A + voters_B], dominant[1]] = 5

    if voters_other > 0 and p > 2:
        minor_parties = [i for i in range(p) if i not in dominant]
        if minor_parties:
            chosen_minors = rng.choice(minor_parties, size=voters_other)
            scores[voter_indices[voters_A + voters_B:], chosen_minors] = 4

    return np.clip(scores, 0, score_scale - 1)


def generate_per_district_data(ballotargs):
    """ Generate per district data """
    (district_candidates, num_voters, dominant, star_pref, party_names,
     num_parties, num_districts, total_domin_pref, rng) = ballotargs

    # Generate all STAR/RRV ballots for all districts at once
    total_voters_nationwide = num_voters * num_districts
    all_star_ballots_np = generate_split_voter_blocks(total_voters_nationwide, dominant, rng, star_pref, num_parties)

    # Generate all FPTP ballots for all districts at once
    all_fptp_ballots_np = np.vstack([
        generate_scores_fptp(num_voters, num_parties, dominant, rng, total_domin_pref)
        for _ in range(num_districts)
    ])

    # Create DataFrames for national tallies and tactical voting
    national_df = pd.DataFrame(all_star_ballots_np, columns=party_names)
    tactical_df = tactical_voting_vectorized(national_df, party_names)

    # --- List PR (Largest Remainder Method/Hare Quota) ---
    total_fptp_voters = num_voters * num_districts
    voters_for_coalition_A = round(total_fptp_voters * 0.5)

    prob_A_major = total_domin_pref / 2
    prob_B_major = total_domin_pref / 2
    prob_minor = (1 - total_domin_pref) / (num_parties - 2) if (num_parties - 2) > 0 else 0

    probs_A = np.zeros(num_parties)
    probs_A[dominant[0]] = prob_A_major
    even_minors = np.arange(2, num_parties, 2)
    probs_A[even_minors] = prob_minor
    if probs_A.sum() > 0: probs_A /= probs_A.sum()

    probs_B = np.zeros(num_parties)
    probs_B[dominant[1]] = prob_B_major
    odd_minors = np.arange(3, num_parties, 2)
    probs_B[odd_minors] = prob_minor
    if probs_B.sum() > 0: probs_B /= probs_B.sum()

    choices_A = rng.choice(party_names, p=probs_A, size=voters_for_coalition_A)
    choices_B = rng.choice(party_names, p=probs_B, size=total_fptp_voters - voters_for_coalition_A)
    single_choice_votes = np.concatenate([choices_A, choices_B])

    return all_star_ballots_np, all_fptp_ballots_np, tactical_df, single_choice_votes


# Tally Votes
def tally_votes(tallyargs):
    " Tally votes based on each voting system """
    (all_star_ballots_np, all_fptp_ballots_np, tactical_df, national_seats,
     num_districts, num_voters, single_choice_votes, thresh, party_names, district_candidates) = tallyargs

    # Filter national STAR/RRV ballots based on threshold
    filtered_df = threshold_filter(tactical_df, thresh)

    # --- STAR + RRV ---
    district_star_wins = pd.Series(0, index=party_names)
    for i, d in enumerate(district_candidates):
        start, end = i * num_voters, (i + 1) * num_voters
        district_df = pd.DataFrame(all_star_ballots_np[start:end], columns=district_candidates[d])
        winner = star_voting_winner(district_df)
        winner_party = winner.split('_')[0]
        district_star_wins[winner_party] += 1

    rrv_seats = reweighted_range_voting(filtered_df, national_seats)

    # --- FPTP + List PR ---
    district_fptp_wins = pd.Series(0, index=party_names)
    for i, d in enumerate(district_candidates):
        start, end = i * num_voters, (i + 1) * num_voters
        district_df = pd.DataFrame(all_fptp_ballots_np[start:end], columns=district_candidates[d])
        winner = fptp_winner(district_df)
        winner_party = winner.split('_')[0]
        district_fptp_wins[winner_party] += 1

    vote_counts = pd.Series(single_choice_votes).value_counts().reindex(party_names, fill_value=0)
    hare_quota_seats = pd.Series(0, index=party_names)
    if not vote_counts.empty and national_seats > 0:
        vote_share = vote_counts / vote_counts.sum()
        eligible_parties = vote_share[vote_share >= thresh].index
        eligible_votes = vote_counts[eligible_parties]

        if eligible_votes.sum() > 0:
            quota = eligible_votes.sum() / national_seats
            initial_seats = (eligible_votes // quota).astype(int)
            remainders = eligible_votes % quota
            seats_to_allocate = int(national_seats - initial_seats.sum())
            remainder_winners = remainders.nlargest(seats_to_allocate).index

            lrm_additional = pd.Series(0, index=eligible_parties)
            lrm_additional.loc[remainder_winners] = 1
            hare_quota_seats = (initial_seats + lrm_additional).reindex(party_names, fill_value=0)

    # --- MMP ---
    total_seats = num_districts + national_seats
    ideal_seats = pd.Series(0, index=party_names)
    if not filtered_df.empty:
        ideal_proportions = filtered_df.sum() / filtered_df.sum().sum()
        ideal_seats = (ideal_proportions * total_seats).round().astype(int)

        seat_diff = total_seats - ideal_seats.sum()
        if seat_diff != 0:
            remainders = (ideal_proportions * total_seats) - ideal_seats
            if seat_diff > 0:
                ideal_seats[remainders.nlargest(int(seat_diff)).index] += 1
            else:
                ideal_seats[remainders.nsmallest(int(abs(seat_diff))).index] -= 1

    mmp_topup = (ideal_seats - district_fptp_wins).clip(lower=0)
    total_mmp = district_fptp_wins + mmp_topup

    return district_star_wins, rrv_seats, district_fptp_wins, hare_quota_seats, total_mmp


# Plotting
def plot_results(plotargs, out_path):
    (thresh, total_domin_pref, domin_star_voter_ratio, num_voters,
     district_star_wins, rrv_seats, district_fptp_wins, hare_quota_seats, total_mmp) = plotargs

    os.makedirs(out_path, exist_ok=True)

    # Combine results for sorting and plotting
    combined_star_rrv = pd.DataFrame({'District (STAR)': district_star_wins, 'National (RRV)': rrv_seats}).fillna(0).astype(int)
    combined_fptp_hare = pd.DataFrame({'District (FPTP)': district_fptp_wins, 'National (Hare Quota)': hare_quota_seats}).fillna(0).astype(int)
    mmp_plot_data = pd.DataFrame({'District (FPTP)': district_fptp_wins, 'Top-up': total_mmp - district_fptp_wins}).fillna(0).astype(int)

    # Sort parties based on STAR+RRV total for consistent ordering
    sort_order = combined_star_rrv.sum(axis=1).sort_values().index

    fig, ax = plt.subplots(1, 3, figsize=(20, 7))

    combined_star_rrv.loc[sort_order].plot(kind='barh', stacked=True, ax=ax[0], color=['#fde047', '#7dd3fc'])
    ax[0].set_title(f"STAR + RRV ({thresh*100:.1f}% Threshold)")
    ax[0].set_xlabel("Seats")

    combined_fptp_hare.loc[sort_order].plot(kind='barh', stacked=True, ax=ax[1], color=['#fca5a5', '#fdba74'])
    ax[1].set_title(f"FPTP + Party List ({thresh*100:.1f}% Threshold)")
    ax[1].set_xlabel("Seats")

    mmp_plot_data.loc[sort_order].plot(kind='barh', stacked=True, ax=ax[2], color=['#bbf7d0', '#86efac'])
    ax[2].set_title(f"MMP ({thresh*100:.1f}% Threshold)")
    ax[2].set_xlabel("Seats")

    chart_title = (f"{num_voters} voters per district; {total_domin_pref*100:.0f}% support for major parties; "
                   f"{domin_star_voter_ratio*100:.0f}% of major party voters score minor higher")
    fig.suptitle(chart_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_path, "result.png"))
    plt.close()


def main(args):
    # Parse arguments
    out_path = args.output
    num_voters = args.voters
    num_parties = args.parties
    num_districts = args.districts
    national_seats = args.national if args.national is not None else num_districts // 2
    total_domin_pref = args.major
    domin_star_voter_ratio = args.star
    thresh = args.threshold

    print("--- Simulation Parameters ---")
    print(f"Voters per District: {num_voters}, Total Voters (approx): {num_voters * num_districts}")

    party_names = [f"Party {chr(65+i)}" for i in range(num_parties)]
    print(f"Parties: {party_names}")

    dominant = [0, 1]
    star_pref = (total_domin_pref * domin_star_voter_ratio) + (1 - total_domin_pref)

    print(f"Support for Major Parties (FPTP): {total_domin_pref:.2%}")
    print(f"Ratio of Major Voters Scoring Minor Higher (STAR): {domin_star_voter_ratio:.2%}")
    print(f"Total Ratio of Voters Scoring Minor Higher (STAR): {star_pref:.2%}")

    rng = np.random.default_rng(seed=42)
    district_candidates = { f"District_{i+1}": [f"{p}_Cand_{i+1}" for p in party_names] for i in range(num_districts) }

    ballotargs = (district_candidates, num_voters, dominant, star_pref, party_names,
                  num_parties, num_districts, total_domin_pref, rng)
    all_star_ballots_np, all_fptp_ballots_np, tactical_df, single_choice_votes = generate_per_district_data(ballotargs)

    tallyargs = (all_star_ballots_np, all_fptp_ballots_np, tactical_df, national_seats,
                 num_districts, num_voters, single_choice_votes, thresh, party_names, district_candidates)
    district_star, rrv, district_fptp, hare_quota, total_mmp = tally_votes(tallyargs)

    print("\n--- Seat Results ---")
    print(f"STAR + RRV: District={district_star.sum()}, National={rrv.sum()}, Total={district_star.sum() + rrv.sum()}")
    print(f"FPTP + List: District={district_fptp.sum()}, National={hare_quota.sum()}, Total={district_fptp.sum() + hare_quota.sum()}")
    print(f"MMP: District={district_fptp.sum()}, Top-up={(total_mmp - district_fptp).sum()}, Total={total_mmp.sum()}")

    plotargs = (thresh, total_domin_pref, domin_star_voter_ratio, num_voters,
                district_star, rrv, district_fptp, hare_quota, total_mmp)
    plot_results(plotargs, out_path)
    print(f"\nChart saved to '{os.path.join(out_path, 'result.png')}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate and compare voting systems.")
    parser.add_argument('-o', '--output', type=str, default='./results/', help='Path for output folder.')
    parser.add_argument('-v', '--voters', type=int, default=1000, help="Number of voters PER DISTRICT.")
    parser.add_argument('-p', '--parties', type=int, default=6, help="Number of parties.")
    parser.add_argument('-d', '--districts', type=int, default=79, help="Number of district seats.")
    parser.add_argument('-n', '--national', type=int, default=None, help="Number of national party list seats. Defaults to half of district seats.")
    parser.add_argument('-m', '--major', type=float, default=0.8, help="Ratio of voters for both major parties in a FPTP election.")
    parser.add_argument('-s', '--star', type=float, default=0.15, help="Ratio of major party voters who would score minor candidates higher in STAR.")
    parser.add_argument('-t', '--threshold', type=float, default=0.02, help="Threshold vote ratio to be eligible for a national seat.")
    parser.add_argument('--save', action='store_true', help="Save generated ballot files (Not implemented in this version).")
    args = parser.parse_args()

    main(args)

