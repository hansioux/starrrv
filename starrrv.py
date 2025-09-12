#!/usr/bin/env python
# coding: utf-8

# # Voting System Simulation: STAR + RRV with Tactical Voting and 2% Threshold
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
    weights = np.ones(len(score_df))
    seats_won = pd.Series(0, index=score_df.columns)
    # The maximum possible score is score_scale - 1 (i.e., 5)
    max_score = score_scale - 1
    for _ in range(seats):
        weighted_scores = (score_df.T * weights).T
        winner = weighted_scores.sum().idxmax()
        seats_won[winner] += 1
        support = score_df[winner] / max_score
        weights /= (1 + support)
    return seats_won


# FPTP voting
def fptp_winner(score_df):
    # For simulation, we assume a voter's single choice in FPTP is for the candidate they scored highest.
    # idxmax(axis=1) finds the column name with the max value for each row.
    return score_df.idxmax(axis=1).value_counts().idxmax()


# Tactical voting: Even → Party A, Odd → Party B
def tactical_voting(df, party_names):
    df = df.copy()
    for i, row in df.iterrows():
        max_party = row.idxmax()
        max_idx = party_names.index(max_party)
        if max_party == "Party A":
            df.at[i, "Party B"] = min(score_scale - 1, df.at[i, "Party B"] + 1)
        elif max_party == "Party B":
            df.at[i, "Party A"] = min(score_scale - 1, df.at[i, "Party A"] + 1)
        elif max_idx >= 2:
            if max_idx % 2 == 0:
                df.at[i, "Party A"] = min(score_scale - 1, df.at[i, "Party A"] + 1)
            else:
                df.at[i, "Party B"] = min(score_scale - 1, df.at[i, "Party B"] + 1)
    return df


# Threshold filtering
def threshold_filter(df, threshold=0.02):
    totals = df.sum()
    share = totals / totals.sum()
    return df[share[share >= threshold].index]

# Generate split block scores
def generate_block_scores(v, pt, rng, dominant, favor_minor):
    '''
    Allocates scores to each of the parties for v voters, based on party and coalition preference.
    '''
    scores = np.zeros((v, pt), dtype=int)
    for i in range(v):
        if favor_minor:
            # Voter prefers a minor party's coalition.
            # Randomly pick Coalition A (even) or B (odd).
            coalition_A_pref = rng.choice([0, 1])

            if coalition_A_pref == 1: # Prefers Coalition A
                scores[i, dominant[0]] = 1 # Low score for major party
                for j in range(2, pt, 2): # High score for even-indexed minor parties
                    scores[i, j] = rng.choice([2, 3, 4, 5])
            else: # Prefers Coalition B
                scores[i, dominant[1]] = 1 # Low score for major party
                for j in range(3, pt, 2): # High score for odd-indexed minor parties
                    scores[i, j] = rng.choice([2, 3, 4, 5])
        else:
            # Voter is a loyal major party supporter.
            # Randomly pick Party A or Party B.
            voter_choice = rng.choice(dominant)

            if voter_choice == dominant[0]: # Voter chooses Party A
                scores[i, dominant[0]] = 5 # Max score for Party A
                # Give lower scores to allied minor parties
                for j in range(2, pt, 2):
                    scores[i, j] = rng.choice([3, 4])
            else: # Voter chooses Party B
                scores[i, dominant[1]] = 5 # Max score for Party B
                # Give lower scores to allied minor parties
                for j in range(3, pt, 2):
                    scores[i, j] = rng.choice([3, 4])

    # Assign low random scores to parties outside the voter's preferred coalition
    low_scores_mask = (scores == 0)
    num_low_scores = np.sum(low_scores_mask)
    scores[low_scores_mask] = rng.choice([0, 1], size=num_low_scores)

    return scores


# def generate_block_scores(v, pt, rng, dominant, favor_minor):
#     '''
#     Allocates scores to each of the parties for v voters, based on party and coalition preference.
#     '''
#     scores = np.zeros((v, pt), dtype=int)
#     for i in range(v):
#         # Determine coalition preference for this voter
#         coalition_A_pref = rng.choice([0, 1])
#         coalition_B_pref = 1 - coalition_A_pref
#
#         if favor_minor:
#             # Voters score minor parties ahead of major parties in their coalition
#             scores[i, dominant[0]] = coalition_A_pref
#             scores[i, dominant[1]] = coalition_B_pref
#             for j in range(2, pt):
#                 if j % 2 == 0 and coalition_A_pref == 1: # Coalition A minor party
#                     scores[i, j] = rng.choice([2, 3, 4, 5])
#                 elif j % 2 != 0 and coalition_B_pref == 1: # Coalition B minor party
#                     scores[i, j] = rng.choice([2, 3, 4, 5])
#                 else:
#                     scores[i, j] = rng.choice([0, 1])
#         else:
#             # Voters score major parties ahead of minor parties
#             scores[i, dominant[0]] = 5 if coalition_A_pref == 1 else 0
#             scores[i, dominant[1]] = 5 if coalition_B_pref == 1 else 0
#             for j in range(2, pt):
#                 if j % 2 == 0 and scores[i, dominant[0]] == 5: # Coalition A minor party
#                     scores[i, j] = rng.choice([3, 4])
#                 elif j % 2 != 0 and scores[i, dominant[1]] == 5: # Coalition B minor party
#                     scores[i, j] = rng.choice([3, 4])
#                 else:
#                     scores[i, j] = rng.choice([0, 1])
#     return scores


def generate_split_voter_blocks(n_voters, dominant, rng, ratio, num_parties):
    '''
    Split voter block based on the ratio of voters that would score minor parties
    ahead of major party in the same coalition.
    '''
    b1_voters = round(n_voters * ratio)
    b2_voters = n_voters - b1_voters
    block1 = generate_block_scores(b1_voters, num_parties, rng, dominant, favor_minor=True)
    block2 = generate_block_scores(b2_voters, num_parties, rng, dominant, favor_minor=False)
    return np.vstack((block1, block2))


# --- NEW AND CORRECTED FUNCTION ---
def generate_scores(n, p, dominant, rng, total_domin_pref):
    """
    Generates scores for an FPTP simulation with a clear 40/40/20 split.
    'n' is number of voters, 'p' is number of parties.
    """
    # 1. Define voter blocks
    voters_A = round(n * (total_domin_pref / 2))
    voters_B = round(n * (total_domin_pref / 2))
    voters_other = n - voters_A - voters_B
    
    # 2. Generate base scores (e.g., random low scores for all)
    scores = rng.integers(0, 3, size=(n, p))

    # 3. Assign strong preference for each block
    # Party A voters give Party A the max score
    scores[0:voters_A, dominant[0]] = 5
    # Party B voters give Party B the max score
    scores[voters_A : voters_A + voters_B, dominant[1]] = 5

    # Optional: Other voters slightly prefer a random minor party
    if voters_other > 0 and p > 2:
        minor_parties = [i for i in range(p) if i not in dominant]
        if minor_parties:
            chosen_minors = rng.choice(minor_parties, size=voters_other)
            rows = np.arange(voters_A + voters_B, n)
            scores[rows, chosen_minors] = 4 # Give a high score, but less than major supporters

    # 4. Shuffle the rows so voter blocks are not sequential
    rng.shuffle(scores)

    return np.clip(scores, 0, score_scale - 1)


def generate_per_district_data(ballotargs):
    """ Generate per district data """
    (district_candidates, num_voters, dominant, star_pref, thresh, party_names,
     num_parties, num_districts, minor_pref, domin_pref, total_domin_pref, rng) = ballotargs

    # Initialize parameters
    district_scores = {}
    national_scores_list = []
    fptp_district_scores = {}

    for d in district_candidates:
        # STAR and RRV ballots are generated with tactical considerations
        district_scores[d] = pd.DataFrame(generate_split_voter_blocks(num_voters, dominant, rng, star_pref, num_parties), columns=district_candidates[d])
        national_scores_list.append(generate_split_voter_blocks(num_voters, dominant, rng, star_pref, num_parties))

        # FPTP ballots are now generated with the corrected function
        fptp_district_scores[d] = pd.DataFrame(generate_scores(num_voters, num_parties, dominant, rng, total_domin_pref), columns=district_candidates[d])

    national_df = pd.DataFrame(np.vstack(national_scores_list), columns=party_names)
    tactical_df = tactical_voting(national_df, party_names)
    filtered_df = threshold_filter(tactical_df, thresh)

    # --- List PR (Largest Remainder Method/Hare Quota) ---
    single_choice_votes = []
    total_fptp_voters = num_voters * num_districts
    voters_for_coalition_A = round(total_fptp_voters * 0.5)
    voters_for_coalition_B = total_fptp_voters - voters_for_coalition_A
    prob_A = total_domin_pref / 2
    prob_B = total_domin_pref - prob_A
    prob_minor = (1 - total_domin_pref) / (num_parties - 2) if (num_parties - 2) > 0 else 0

    probs_A = np.zeros(num_parties)
    for i in range(num_parties):
        if i == dominant[0]: probs_A[i] = prob_A
        elif i not in dominant and i % 2 == 0: probs_A[i] = prob_minor
    if probs_A.sum() > 0: probs_A /= probs_A.sum()

    probs_B = np.zeros(num_parties)
    for i in range(num_parties):
        if i == dominant[1]: probs_B[i] = prob_B
        elif i not in dominant and i % 2 != 0: probs_B[i] = prob_minor
    if probs_B.sum() > 0: probs_B /= probs_B.sum()

    choices_A = rng.choice(party_names, p=probs_A, size=voters_for_coalition_A)
    choices_B = rng.choice(party_names, p=probs_B, size=voters_for_coalition_B)
    single_choice_votes.extend(choices_A)
    single_choice_votes.extend(choices_B)

    return district_scores, fptp_district_scores, national_df, tactical_df, filtered_df, single_choice_votes


# Tally Votes
def tally_votes(tallyargs):
    " Tally votes based on each voting system """
    (district_scores, national_df, national_seats, fptp_district_scores,
     filtered_df, num_districts, single_choice_votes, thresh, party_names) = tallyargs

    # STAR district winners
    district_star = pd.Series(0, index=party_names)
    for df in district_scores.values():
        winner = star_voting_winner(df)
        for p in party_names:
            if winner.startswith(p):
                district_star[p] += 1
                break

    # Proportional allocations
    rrv = reweighted_range_voting(national_df, national_seats)

    # Combine for stacked STAR + RRV chart
    combined_star_rrv = pd.DataFrame({
        'District (STAR)': district_star,
        'National (RRV)': rrv
    }).fillna(0).astype(int)

    # FPTP district winners
    district_fptp = pd.Series(0, index=party_names)
    for df in fptp_district_scores.values():
        winner = fptp_winner(df)
        for p in party_names:
            if winner.startswith(p):
                district_fptp[p] += 1
                break

    # --- List PR (Largest Remainder Method/Hare Quota) ---
    vote_counts = pd.Series(single_choice_votes).value_counts().reindex(party_names, fill_value=0)
    total_valid_votes = vote_counts.sum()
    vote_share = vote_counts / total_valid_votes
    eligible_parties = vote_share[vote_share >= thresh].index
    eligible_votes = vote_counts[eligible_parties]

    if not eligible_votes.empty and national_seats > 0:
        quota = eligible_votes.sum() / national_seats
        initial_seats = (eligible_votes // quota).astype(int)
        remainders = eligible_votes % quota
        seats_to_allocate = int(national_seats - initial_seats.sum())
        remainder_winners = remainders.nlargest(seats_to_allocate).index
        lrm_additional = pd.Series(0, index=party_names)
        lrm_additional.loc[remainder_winners] = 1
        hare_quota_seats = initial_seats.reindex(party_names, fill_value=0) + lrm_additional
    else:
        hare_quota_seats = pd.Series(0, index=party_names)

    # MMP results
    ideal_scores = filtered_df.sum()
    total_seats = num_districts + national_seats
    if ideal_scores.sum() > 0:
        ideal_proportions = ideal_scores / ideal_scores.sum() * total_seats
        ideal_seats = ideal_proportions.round().astype(int)
        # Adjust if total seats don't match due to rounding
        seat_diff = total_seats - ideal_seats.sum()
        if seat_diff != 0:
            remainders = ideal_proportions - ideal_seats
            if seat_diff > 0:
                ideal_seats[remainders.nlargest(int(seat_diff)).index] += 1
            else:
                ideal_seats[remainders.nsmallest(int(abs(seat_diff))).index] -= 1
    else:
        ideal_seats = pd.Series(0, index=party_names)


    mmp_topup = ideal_seats - district_fptp
    mmp_topup[mmp_topup < 0] = 0
    total_mmp = district_fptp + mmp_topup

    return district_star, rrv,  combined_star_rrv, district_fptp, hare_quota_seats, total_mmp


# Plotting
def plot_results(plotargs, out_path):
    (thresh, total_domin_pref, domin_star_voter_ratio, num_voters,
     combined_star_rrv, district_fptp, hare_quota_seats, total_mmp) = plotargs

    os.makedirs(out_path, exist_ok=True)
    th_int = str(int(thresh*100))
    star_title = f"STAR + RRV ({th_int}% Threshold, Tactical, Minor-Friendly)"
    fptp_title = f"FPTP + Party List ({th_int}% Threshold)"
    mmp_title = f"MMP ({th_int}% Threshold)"
    tdp_int = str(int(total_domin_pref*100))
    ds_int = str(int(domin_star_voter_ratio *100))
    chart_title = (f"{num_voters} voters per district; {tdp_int}% support for major parties; "
                   f"{ds_int}% of major party voters would score aligned minor party higher.")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: STAR + RRV
    combined_star_rrv.sort_values(by=["District (STAR)", "National (RRV)"], ascending=True).plot(
        kind='barh', stacked=True, ax=ax[0], color=['gold', 'skyblue']
    )
    ax[0].legend(loc='lower right')
    ax[0].set_title(star_title)
    ax[0].set_xlabel("Seats")

    # Plot 2: FPTP + Hare Quota
    combined_fptp_harelist = pd.DataFrame({
        'District (FPTP)': district_fptp,
        'National (Hare quota)': hare_quota_seats
    }).fillna(0).astype(int)

    combined_fptp_harelist.sort_values(by=["District (FPTP)", "National (Hare quota)"], ascending=True).plot(
        kind='barh', stacked=True, ax=ax[1], color=['salmon', 'orange']
    )
    ax[1].legend(loc='lower right')
    ax[1].set_title(fptp_title)
    ax[1].set_xlabel("Seats")

    # Plot 3: MMP
    mmp_plot_data = pd.DataFrame({
        'District (FPTP)': district_fptp,
        'Top-up': total_mmp - district_fptp
    }).fillna(0).astype(int)
    mmp_plot_data.sort_values(by=['District (FPTP)', 'Top-up'], ascending=True).plot(
        kind='barh', stacked=True, ax=ax[2], color=['lightgreen', 'darkseagreen']
    )
    ax[2].set_title(mmp_title)
    ax[2].set_xlabel("Seats")

    fig.suptitle(chart_title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(out_path, "result.png"))
    plt.close()


# Save ballots
def save_ballots(saveargs, out_path):
    (num_voters, num_districts, national_seats, num_parties, domin_pref,
    domin_star_voter_ratio, district_scores, national_df, fptp_district_scores,
    single_choice_votes) = saveargs

    spath = os.path.join(out_path, f"{num_voters}_{num_districts}_{national_seats}_{num_parties}_{domin_pref:.2f}_{domin_star_voter_ratio:.2f}")
    os.makedirs(spath, exist_ok=True)

    # STAR district scores
    for df_name, df in district_scores.items():
        filename = os.path.join(spath, f"star_{df_name}.csv")
        df.to_csv(filename, index=False)
    national_df.to_csv(os.path.join(spath, 'rrv_ballots.csv'), index=False)

    # FPTP district scores
    for df_name, df in fptp_district_scores.items():
        filename = os.path.join(spath, f"fptp_{df_name}.csv")
        df.to_csv(filename, index=False)
    # FPTP Party list votes
    np.savetxt(os.path.join(spath, 'fptp_partylist.csv'), single_choice_votes, fmt="%s", delimiter=",")


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

    print("Number of voters per district:", num_voters)
    print("Total voters (approx):", num_voters * num_districts)

    # Generate party names and party data
    party_names = [f"Party {chr(65+i)}" for i in range(num_parties)]
    print("Parties:", party_names)

    dominant = [0, 1]
    minor = [x for x in range(num_parties) if x not in dominant]

    # Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed=42)

    district_candidates = {
        f"District_{i+1}": [f"{party_names[j]}_Cand_{i+1}" for j in range(num_parties)]
        for i in range(num_districts)
    }

    # Calculate voter preference ratios
    domin_pref = total_domin_pref / len(dominant)
    minor_pref = (1 - total_domin_pref) / len(minor) if len(minor) > 0 else 0
    star_pref = (domin_pref * domin_star_voter_ratio * len(dominant)) + (minor_pref * len(minor))

    print(f"Support for major parties in FPTP: {total_domin_pref:.2%}")
    print(f"Support for a major party: {domin_pref:.2%}")
    print(f"Support for a minor party: {minor_pref:.2%}")
    print(f"Ratio of major party voters scoring aligned minor party higher in STAR/RRV: {domin_star_voter_ratio:.2%}")
    print(f"Total ratio of voters scoring minor parties higher: {star_pref:.2%}")

    # Fool-proof ratio parameters
    if any(v > 1 for v in [total_domin_pref, domin_pref, minor_pref, domin_star_voter_ratio, star_pref]):
        print("\n[Error] Ratio cannot be larger than 100%")
        return

    # Generate ballots
    ballotargs = (district_candidates, num_voters, dominant, star_pref, thresh,
                  party_names, num_parties, num_districts, minor_pref, domin_pref, total_domin_pref, rng)
    district_scores, fptp_district_scores, national_df, tactical_df, filtered_df, single_choice_votes = generate_per_district_data(ballotargs)

    # Tally votes
    tallyargs = (district_scores, national_df, national_seats,
                 fptp_district_scores, filtered_df, num_districts,
                 single_choice_votes, thresh, party_names)
    district_star, rrv, combined_star_rrv, district_fptp, hare_quota_seats, total_mmp = tally_votes(tallyargs)

    print("\n--- Seat Results ---")
    print(f"Total STAR + RRV seats: {combined_star_rrv.sum().sum()} (District: {district_star.sum()}, National: {rrv.sum()})")
    print(f"Total FPTP + List PR seats: {district_fptp.sum() + hare_quota_seats.sum()} (District: {district_fptp.sum()}, National: {hare_quota_seats.sum()})")
    print(f"Total MMP seats: {total_mmp.sum()} (District: {district_fptp.sum()}, Top-up: {(total_mmp - district_fptp).sum()})")

    # Plot results
    plotargs = (thresh, total_domin_pref, domin_star_voter_ratio, num_voters,
                combined_star_rrv, district_fptp, hare_quota_seats, total_mmp)
    plot_results(plotargs, out_path)

    # Save ballots
    if args.save:
        saveargs = (num_voters, num_districts, national_seats, num_parties, domin_pref,
                    domin_star_voter_ratio, district_scores, national_df, fptp_district_scores,
                    single_choice_votes)
        save__ballots(saveargs, out_path)
        print(f"\nBallots and results saved to '{out_path}' directory.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate and compare voting systems.")
    parser.add_argument('-b', '--ballots', type=str, required=False, help='Input folder with saved ballots in csv format')
    parser.add_argument('-c', '--config', type=str, required=False, help='Path to parameter config file')
    parser.add_argument('-o', '--output', type=str, default='./results/', required=False, help='Path for output folder')
    parser.add_argument('-v', '--voters', type=int, default=1000, required=False, help="Number of voters PER DISTRICT.")
    parser.add_argument('-p', '--parties', type=int, default=6, required=False, help="Number of parties.")
    parser.add_argument('-d', '--districts', type=int, default=79, required=False, help="Number of district seats.")
    parser.add_argument('-n', '--national', type=int, required=False, help="Number of national party list seats. Defaults to half of district seats.")
    parser.add_argument('-m', '--major', type=float, default=0.8, required=False, help="Ratio of voters for both major parties in a FPTP election.")
    parser.add_argument('-s', '--star', type=float, default=0.15, required=False, help="Ratio of major party voters who would score minor candidates higher in STAR.")
    parser.add_argument('-t', '--threshold', type=float, default=0.02, required=False, help="Threshhold vote ratio to be eligible for a national seat.")
    parser.add_argument('--save', action='store_true', help="Save generated ballot files.")
    args = parser.parse_args()

    main(args)
