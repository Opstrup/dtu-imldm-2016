# Plot results
BIC = np.loadtxt("gmm_cv_BIC_0-40.txt")
np.savetxt("gmm_cv_AIC_40-60.txt", AIC)
np.savetxt("gmm_cv_CVE_40-60.txt", CVE)
figure(1); hold(True)
plot(KRange, BIC)
plot(KRange, AIC)
plot(KRange, 2*CVE)
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()
